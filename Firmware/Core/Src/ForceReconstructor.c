//Porting Force Reconstructor class in C 

#include "ForceReconstructor.h"

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

//init della struct
void ForceReconstructor_Init(ForceReconstructor *fr)
{
    memset(fr, 0, sizeof(ForceReconstructor)); //Azzero la struct così so che negli array ho valori noti
    
    fr->NW = 500;
    fr->Thr_samples = 100;
    fr->warmup_samples = 250;
    fr->press_sigma = 10.0f;
    fr->press_confirm = 5;
    fr->alpha = 0.3f;
    fr->nSamples_adaptive_offset = 50;
    fr->samples_after_release = 250;
    fr->signal2noise_ratio = 10.0f;
    fr->min_press_sensors = 2;
    fr->release_ratio = 0.5f;
    fr->n_sensors = 8; 
}

/* 
 * Funzione di supporto per la funzione qsort() di libreria standard C.
 * Serve per ordinare un array di numeri float in ordine crescente.
 * Ritorna:
 *   1  se a > b
 *  -1  se a < b
 *   0  se a == b
 */
static int compare_floats(const void *a, const void *b)
{
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

//------- CALCOLO SOGLIE DINAMICHE -------
void compute_thresholds(ForceReconstructor *fr)
{
    //Buffer temporanei per estrarre e manipolare i dati di un singolo sensore alla volta
    float column_buffer[THR_SAMPLES]; 
    float abs_diff_buffer[THR_SAMPLES];

    for(uint8_t ns = 0; ns < fr->n_sensors; ns++)
    {
        //estrazione colonna del rumore per il sensore ns
        for(uint16_t i = 0; i < fr->Thr_samples; i++)
        {
            column_buffer[i] = fr->noise_buffer[i][ns];
        }

        //------- Mediana -------
        qsort(column_buffer, fr->Thr_samples, sizeof(float), compare_floats); //ordiniamo i campioni dal più piccolo al più grande

        float median;
        if(fr->Thr_samples % 2 == 0) // Se il numero di campioni è pari, la mediana è la media dei due elementi centrali
            median = (column_buffer[fr->Thr_samples / 2 - 1] + column_buffer[fr->Thr_samples / 2]) / 2.0f;
        else // Se dispari, è l'elemento perfettamente al centro
            median = column_buffer[fr->Thr_samples / 2];

        //------- Deviazione Standard (sigma) -------
        //media della colonna del rumore
        float sum = 0.0f;
        for(uint16_t i = 0; i < fr->Thr_samples; i++)
        {
            sum += fr->noise_buffer[i][ns];
        }
        float mean = sum / (float)fr->Thr_samples;

        //somma dei quadrati degli scarti rispetto alla media
        float sum_sq_diff = 0.0f;
        for(uint16_t i = 0; i < fr->Thr_samples; i++)
        {
            float diff = fr->noise_buffer[i][ns] - mean;
            sum_sq_diff += diff * diff;
        }

        //deviazione std campionaria (radice quadrata diviso N-1)
        float sigma = sqrtf(sum_sq_diff / (float)(fr->Thr_samples - 1));

        //soglia di pressione (in python: self.thr_press = self.press_sigma * sigma)
        fr->thr_press[ns] = fr->press_sigma * sigma;

        //------- Max Noise -------
        //calcolo differenza assoluta di ogni campione della mediana
        for (uint16_t i = 0; i < fr->Thr_samples; i++) {
            abs_diff_buffer[i] = fabsf(fr->noise_buffer[i][ns] - median);
        }

        // Ordiniamo le deviazioni assolute in ordine crescente
        qsort(abs_diff_buffer, fr->Thr_samples, sizeof(float), compare_floats);

        // Troviamo l'indice che delimita il 99° percentile
        uint16_t idx_99 = (uint16_t)(0.99f * fr->Thr_samples);
        float sum_top_noise = 0.0f;
        uint16_t count_top_noise = 0;

        //Sommiamo solo gli elementi dal percentile 99 in poi (gli ultimi campioni più alti)
        for (uint16_t i = idx_99; i < fr->Thr_samples; i++) {
            sum_top_noise += abs_diff_buffer[i];
            count_top_noise++;
        }

        //Facciamo la media dei campioni selezionati
        if (count_top_noise > 0) {
            fr->max_noise[ns] = sum_top_noise / (float)count_top_noise;
        } else {
            fr->max_noise[ns] = 0.0f;
        }

        //------- Offset Adattivo -------
        // Troviamo l'indice di partenza degli ultimi 'nSamples_adaptive_offset' campioni
        uint16_t start_idx = fr->Thr_samples - fr->nSamples_adaptive_offset;
        float offset_sum = 0.0f;

        for(uint16_t i=0; i < fr->nSamples_adaptive_offset; i++)
        {
            float sample_val = fr->noise_buffer[start_idx + i][ns]; //estraggo il campione recente di rumore
            fr->offset_buffer[i][ns] = sample_val; //salvo il campione nel buffer di offset locale
            offset_sum += sample_val; //sommo i valori per calcolare la media
        }

        //calcolo media ultimi campioni (DC offset)
        fr->adaptive_offset[ns] = offset_sum / (float)fr->nSamples_adaptive_offset;
    }
    
}

//Reset dello stato di un singolo sensore (Python: def _reset_local(self, ns))
static void reset_local(ForceReconstructor *fr, uint8_t ns)
{
    fr->phase[ns] = 0;
    fr->integral[ns] = 0.0f;
    fr->counter[ns] = 0;
    fr->confirm[ns] = 0;
    fr->guard_counter[ns] = fr->samples_after_release;
}

//Reset globale di tutti i sensori e del sistema (Python: def _reset_all_global(self))
static void reset_all_global(ForceReconstructor *fr)
{
    for(uint8_t ns = 0; ns < fr->n_sensors; ns++)
    {
        reset_local(fr, ns);
        fr->press_local_mask[ns] = 0;
        fr->release_local_mask[ns] = 0;
    }

    fr->global_press_active = false;
    fr->global_phase = 0;
    fr->global_press_counter = 0;
}

//------- INTEGRAL STEP -------
void integral_step(ForceReconstructor *fr, const float *x_raw)
{
    memset(fr->current_output, 0, sizeof(fr->current_output)); //reset output corrente

    //------- Warm-up -------
    if(fr->warmup_counter < fr->warmup_samples)
    {
        fr->warmup_counter++;

        memcpy(fr->data_raw_prec, x_raw, sizeof(float) * fr->n_sensors); //salva l'ultimo campione grezzo (serve per inizializzare la memoria del filtro)
        return;
    }

    //------- Noise Learning -------
    if(fr->noise_counter < fr->Thr_samples)
    {
        //salviamo il campione nella riga corrente del noise buffer
        for(uint8_t ns = 0; ns < fr->n_sensors; ns++)
        {
            fr->noise_buffer[fr->noise_counter][ns] = x_raw[ns];
        }
        fr->noise_counter++;

        //quando il buffer è pieno calcolo le soglie
        if(fr->noise_counter == fr->Thr_samples)
        {
            compute_thresholds(fr);
            memcpy(fr->data_raw_prec, x_raw, sizeof(float) * fr->n_sensors);
        }
        return;
    }

    //------- Global Press Timing -------
    if(fr->global_press_active && fr->global_phase == 0)
    {
        fr->global_press_counter++;

        //evita ri attivazioni dopo una pressione validata
        if(fr->global_press_counter >= (uint16_t)roundf(fr->NW * 0.25f))
        {
            fr->global_phase = 1;
        }
    }

    //------- Ciclo per ogni Sensore -------
    for(uint8_t ns = 0; ns < fr->n_sensors; ns++)
    {
        //------- Filtro IIR passa basso exp -------
        float sm = fr->alpha * x_raw[ns] + (1.0f - fr->alpha) * fr->data_raw_prec[ns];
        fr->data_raw_prec[ns] = sm; //salvo valore filtrato per il prossimo ciclo
        float x = sm - fr->adaptive_offset[ns]; //rimuovo offset DC

        //------- Timer Anti Rimbalzo(Guard) -------
        if(fr->guard_counter[ns] > 0)
        {
            fr->guard_counter[ns]--;
            continue; //salta analisi dello stato per questo sensore
        }

        //------- FSM -------
        //======= IDLE =======
        if(fr->phase[ns] == 0)
        {
            if(fr->global_phase == 1)
            {
                continue; //blocco se la pressione globale è già in hold
            }

            //verefico se il segnale supera la soglia del rumore
            if(fabsf(x) > fr->thr_press[ns])
            {
                fr->confirm[ns]++;
            }else
            {
                fr->confirm[ns] = 0;
            }

            //verifico press confirm di campioni consecutivi
            if(fr->confirm[ns] >= fr->press_confirm)
            {
                fr->phase[ns] = 1; //IDLE->PRESS
                fr->integral[ns] = 0.0f;
                fr->counter[ns] = 0;
                fr->confirm[ns] = 0;
            }
            continue;
        }

        //======= PRESS =======
        if(fr->phase[ns] == 1)
        {
            fr->integral[ns] += fabsf(x); //Accumulo forza

            //aggiorno valore massimo dell'integrale
            if(fr->integral[ns] > fr->max_integral[ns])
            {
                fr->max_integral[ns] = fr->integral[ns];
            }

            fr->counter[ns]++;
            fr->press_local_mask[ns] = 1; //marca come premuto

            //controllo per attivare la pressione globale
            if(!fr->global_press_active)
            {
                uint8_t n_press = 0;
                for(uint8_t i = 0; i < fr->n_sensors; i++)
                {
                    if(fr->press_local_mask[i]) n_press++;
                }

                if(n_press >= fr->min_press_sensors)
                {
                    fr->global_press_active = true;
                }
            }

            //Fine finestra integrazione NW
            if(fr->counter[ns] >= fr->NW)
            {
                if(!fr->global_press_active)
                {
                    reset_all_global(fr); //pressione isolata o rumore->reset
                    fr->phase[ns] = 0;
                }else
                {
                    fr->phase[ns] = 2; //PRESS->HOLD
                }
            }
            continue;
        }

        //======= HOLD =======
        if(fr->phase[ns] == 2)
        {
            //cerco il rilascio
            if(fabsf(x) > fr->thr_press[ns])
            {
                fr->confirm[ns]++;
            }else
            {
                fr->confirm[ns] = 0;
            }

            if(fr->confirm[ns] >= fr->press_confirm)
            {
                fr->phase[ns] = 3; //HOLD->RELEASE
                fr->confirm[ns] = 0;
            }
            continue;
        }

        //======= RELEASE =======
        if(fr->phase[ns] == 3)
        {
            fr->integral[ns] -= fabsf(x); //scarico integrale

            //se la forza stimata è scesa sotto il 50% del max
            if(fr->integral[ns] <= 0.5f * fr->max_integral[ns])
            {
                fr->release_local_mask[ns] = 1;
                reset_local(fr, ns); //reset locale e avvio guard time
            }

            //controllo rilascio globale
            uint8_t n_press = 0;
            uint8_t n_release = 0;

            for(uint8_t i = 0; i < fr->n_sensors; i++)
            {
                if(fr->press_local_mask[i]) n_press++;
                if(fr->release_local_mask[i]) n_release++;
            }

            if(n_press > 0 && ((float)n_release / (float)n_press) >= fr->release_ratio)
            {
                reset_all_global(fr);
            }
            continue;
        }
    }//fine loop sensori

    //------- Global Output -------
    if(fr->global_press_active)
    {
        for(uint8_t ns = 0; ns < fr->n_sensors; ns++)
        {
            if(fr->max_integral[ns] > 1e-9f) //evito la divisione per zero controllando con una soglia minima 1e-9
            {
                fr->current_output[ns] = fr->integral[ns] / fr->max_integral[ns];
            }
        }
    }

}

//Ricalibrazione soglie
void recalibrate_thresholds(ForceReconstructor *fr)
{
    //reset contatori di calibrazione
    fr->noise_counter = 0;
    fr->warmup_counter = 0;

    //azzero buffer rumore, offset, memoria segnale
    memset(fr->noise_buffer, 0, sizeof(fr->noise_buffer));
    memset(fr->offset_buffer, 0, sizeof(fr->offset_buffer));
    memset(fr->adaptive_offset, 0, sizeof(fr->adaptive_offset));
    memset(fr->data_raw_prec, 0, sizeof(fr->data_raw_prec));

    //reset stati globali e locali
    reset_all_global(fr);
}

//Reset All
void reset_all(ForceReconstructor *fr)
{
    //reset completo dello stato globale e del max integral
    reset_all_global(fr);
    memset(fr->max_integral, 0, sizeof(fr->max_integral));

    //riavvia calibrazione soglie
    recalibrate_thresholds(fr);
}

void process_block(ForceReconstructor *fr, const float *block_input, float *out_matrix, uint16_t n_samples)
{
    for(uint16_t i = 0; i < n_samples; i++)
    {
        const float *current_sample = &block_input[i * fr->n_sensors]; //puntatore alla riga i esima del blocco di ingresso
        integral_step(fr, current_sample); //eseguo integrazione

        //se utente ha passato buffer di output, copia i risultati
        if(out_matrix != NULL)
        {
            memcpy(&out_matrix[i * fr->n_sensors], fr->current_output, sizeof(float) * fr->n_sensors);
        }
    }
}