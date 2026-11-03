
#ifndef FORCE_RECONSTRUCTOR_H
#define FORCE_RECONSTRUCTOR_H

#include <stdint.h>
#include <stdbool.h>

#define N_SENSORS 8
#define THR_SAMPLES 1500
#define OFFSET_SAMPLES 50


typedef struct 
{
    //------- Parametri -------
    uint16_t NW; 
    uint16_t Thr_samples; 
    uint16_t warmup_samples;
    float press_sigma;
    uint8_t press_confirm;
    float alpha;
    uint16_t nSamples_adaptive_offset;
    uint16_t samples_after_release;
    float signal2noise_ratio;
    uint8_t min_press_sensors;
    float release_ratio;

    //------- Noise -------
    uint16_t warmup_counter;
    uint16_t noise_counter;
    float noise_buffer[THR_SAMPLES][N_SENSORS];

    //------- Adaptive Offset -------
    float offset_buffer[OFFSET_SAMPLES][N_SENSORS];
    float adaptive_offset[N_SENSORS];

    //------- Thresholds -------
    float thr_press[N_SENSORS];
    float max_noise[N_SENSORS];

    //------- Local State -------
    uint8_t phase[N_SENSORS];
    float integral[N_SENSORS];
    float max_integral[N_SENSORS];
    uint16_t counter[N_SENSORS];
    uint8_t confirm[N_SENSORS];
    uint16_t guard_counter[N_SENSORS];

    //------- Global State -------
    bool global_press_active;
    uint8_t global_phase;
    uint16_t global_press_counter;
    uint8_t press_local_mask[N_SENSORS];
    uint8_t release_local_mask[N_SENSORS];

    //------- Signal Memory -------
    float data_raw_prec[N_SENSORS];

    //------- Output -------
    float current_output[N_SENSORS];

    //------- Number of Sensors -------
    uint8_t n_sensors;

}ForceReconstructor;

void ForceReconstructor_Init(ForceReconstructor *fr);
void compute_thresholds(ForceReconstructor *fr);
void integral_step(ForceReconstructor *fr, const float *x_raw);
void recalibrate_thresholds(ForceReconstructor *fr);
void reset_all(ForceReconstructor *fr);
void process_block(ForceReconstructor *fr, const float *block_input, float *out_matrix, uint16_t n_samples);

#endif