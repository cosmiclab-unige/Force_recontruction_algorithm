/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/*LEGENDA CONFIGURAZIONE PARAMETRI VIA UART*/
/**
 * "start o start,1-n_sensors" = avvio acquisizione con n_sensors selezionati
 * "stop" = stop acquisizione
 * "r" = reset leggero, ricalibrazione soglie, MSF, inetgral, max integral
 * "rd" = reset ai parametri di default del ricostruttore, ricalibrazione soglie, MSF, inetgral, max integral
 * "set,PARAMETRO,VALORE" modalità di settaggio dei parametri del ricostruttore
 * "nw" = NW
 * "thr" = Thr_samples
 * "wup" = warmup_samples
 * "pc" = press_confirm
 * "ao" = nSamples_adaptive_offset
 * "sar" = samples_after_release
 * "mp" = min_press_sensors
 * "ns" = n_sensors
 * "sig" = press_sigma
 * "a" = alpha
 * "s2n" = signal2noise_ratio
 * "rr" =release_ratio
 */

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include <stdio.h>
#include "ForceReconstructor.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#define NUM_CHANNELS 64
#define NUM_OF_DATA 32
#define HALF_NUM_OF_DATA 16
#define DIM_TEST_PACKET 8
#define MAX_SELECTED_CHANNELS NUM_CHANNELS
#define PACKET_SIZE 64
#define OFFSET 0
typedef struct __attribute__((packed)) {
    uint8_t  header[3];      // 0x3C, 0x3E, 0x00
    uint8_t n_channels;
    uint32_t timestamp_us;   // from TIM5
    int16_t  ch[PACKET_SIZE];         // channels
    uint16_t  force[N_SENSORS];      // force values
} SamplePacket;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

SPI_HandleTypeDef hspi1;
SPI_HandleTypeDef hspi3;
DMA_HandleTypeDef hdma_spi1_rx;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim5;

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_tx;
DMA_HandleTypeDef hdma_usart1_rx;

/* USER CODE BEGIN PV */
// --- System state flags ---
volatile uint8_t delay_counter = 0;		// Used during the TIM2 startup sequence (test/init logic)
volatile uint8_t acq_enabled   = 0;      // Acquisition enable/disable flag (controlled via UART commands)
volatile uint8_t uart_tx_done  = 1;      // UART DMA transmission ready flag
volatile uint16_t old_pos = 0;

// --- SPI Data Buffers ---
volatile uint16_t rx_spi_buffer[NUM_OF_DATA];   // Buffer used only for the initial test pattern sequence
volatile uint16_t spi_rx_buffer[NUM_CHANNELS];  // Main SPI data buffer (64 channels), filled by DMA

// --- Test pattern verification buffers ---
uint8_t test_packet_1[DIM_TEST_PACKET];         // Extracted bytes from the first half of rx_spi_buffer
uint8_t test_packet_2[DIM_TEST_PACKET];         // Extracted bytes from the second half of rx_spi_buffer

// Reference test packet used to validate ADC test mode output
static const uint8_t ref_packet[DIM_TEST_PACKET] =
    {0x30,0x00,0x66,0xF0,0x24,0x01,0xF6,0x80};

// --- DDC232 Configuration Command ---
static const uint16_t value = 0xEC0;    // 12-bit configuration command sent via SPI3

// --- UART Transmission Packet ---
static SamplePacket pkt;                // DMA-safe data packet (static to avoid stack corruption)

// --- UART Receive Buffers ---
#define CMD_BUFFER_SIZE 128
uint8_t uart_rx_dma_buffer[CMD_BUFFER_SIZE];
char command_buffer[CMD_BUFFER_SIZE];

uint8_t channel_mask[NUM_CHANNELS];   // 1 = canale attivo
uint8_t num_active_channels = 0;

// --- Force Reconstructor ---
ForceReconstructor force_reconstructor;
float force_input[N_SENSORS];
static uint8_t tx_buffer[sizeof(SamplePacket)]; // Buffer for UART transmission (DMA-safe)
static uint16_t spi_rx_snapshot[NUM_CHANNELS]; // Snapshot of SPI data for processing (DMA-safe)
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_SPI3_Init(void);
static void MX_SPI1_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM5_Init(void);
/* USER CODE BEGIN PFP */
static void enable_all_channels(void);
void parse_start_command(char *cmd);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM2_Init();
  MX_USART1_UART_Init();
  MX_SPI3_Init();
  MX_SPI1_Init();
  MX_TIM1_Init();
  MX_TIM3_Init();
  MX_TIM5_Init();
  ForceReconstructor_Init(&force_reconstructor);
  /* USER CODE BEGIN 2 */
  HAL_UART_Receive_DMA(&huart1, uart_rx_dma_buffer, CMD_BUFFER_SIZE);
  __HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);
  HAL_TIM_Base_Start(&htim5);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 120;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SPI3|RCC_PERIPHCLK_SPI1;
  PeriphClkInitStruct.PLL2.PLL2M = 8;
  PeriphClkInitStruct.PLL2.PLL2N = 64;
  PeriphClkInitStruct.PLL2.PLL2P = 16;
  PeriphClkInitStruct.PLL2.PLL2Q = 2;
  PeriphClkInitStruct.PLL2.PLL2R = 2;
  PeriphClkInitStruct.PLL2.PLL2RGE = RCC_PLL2VCIRANGE_3;
  PeriphClkInitStruct.PLL2.PLL2VCOSEL = RCC_PLL2VCOWIDE;
  PeriphClkInitStruct.PLL2.PLL2FRACN = 0;
  PeriphClkInitStruct.Spi123ClockSelection = RCC_SPI123CLKSOURCE_PLL2;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES_RXONLY;
  hspi1.Init.DataSize = SPI_DATASIZE_16BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 0x0;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  hspi1.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
  hspi1.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
  hspi1.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi1.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi1.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
  hspi1.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
  hspi1.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
  hspi1.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
  hspi1.Init.IOSwap = SPI_IO_SWAP_DISABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief SPI3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI3_Init(void)
{

  /* USER CODE BEGIN SPI3_Init 0 */

  /* USER CODE END SPI3_Init 0 */

  /* USER CODE BEGIN SPI3_Init 1 */

  /* USER CODE END SPI3_Init 1 */
  /* SPI3 parameter configuration*/
  hspi3.Instance = SPI3;
  hspi3.Init.Mode = SPI_MODE_MASTER;
  hspi3.Init.Direction = SPI_DIRECTION_1LINE;
  hspi3.Init.DataSize = SPI_DATASIZE_12BIT;
  hspi3.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi3.Init.CLKPhase = SPI_PHASE_2EDGE;
  hspi3.Init.NSS = SPI_NSS_SOFT;
  hspi3.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4;
  hspi3.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi3.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi3.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi3.Init.CRCPolynomial = 0x0;
  hspi3.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  hspi3.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
  hspi3.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
  hspi3.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi3.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi3.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
  hspi3.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
  hspi3.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
  hspi3.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
  hspi3.Init.IOSwap = SPI_IO_SWAP_DISABLE;
  if (HAL_SPI_Init(&hspi3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI3_Init 2 */

  /* USER CODE END SPI3_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 11;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_ENABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 6;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.BreakFilter = 0;
  sBreakDeadTimeConfig.Break2State = TIM_BREAK2_DISABLE;
  sBreakDeadTimeConfig.Break2Polarity = TIM_BREAK2POLARITY_HIGH;
  sBreakDeadTimeConfig.Break2Filter = 0;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 23;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 9;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_SlaveConfigTypeDef sSlaveConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 19;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sSlaveConfig.SlaveMode = TIM_SLAVEMODE_GATED;
  sSlaveConfig.InputTrigger = TIM_TS_ITR0;
  if (HAL_TIM_SlaveConfigSynchro(&htim3, &sSlaveConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 239;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 4294967295;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim5, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 3000000;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);
  /* DMA1_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream1_IRQn);
  /* DMA1_Stream2_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream2_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream2_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_11, GPIO_PIN_SET);

  /*Configure GPIO pin : PD11 */
  GPIO_InitStruct.Pin = GPIO_PIN_11;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : PD12 */
  GPIO_InitStruct.Pin = GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

  /* USER CODE BEGIN MX_GPIO_Init_2 */
  /* UART1 */
  GPIO_InitStruct.Pin = GPIO_PIN_9 | GPIO_PIN_10;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* CONV ON PC7 */
  GPIO_InitStruct.Pin = GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF2_TIM3;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /* CLK ON PE14 */
  GPIO_InitStruct.Pin = GPIO_PIN_14;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF13_TIM1;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);



  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance != TIM2) return;

    switch(delay_counter)
    {
        case 1:
        {
        	HAL_GPIO_WritePin(GPIOD, GPIO_PIN_11, GPIO_PIN_SET);
        	break;
        }
        case 3:
        {
        	HAL_SPI_Transmit(&hspi3, (uint8_t*)&value, 1, HAL_MAX_DELAY);
        	break;
        }
        case 6:
        {
        	HAL_SPI_Receive(&hspi1, (uint8_t*)&rx_spi_buffer, NUM_OF_DATA, HAL_MAX_DELAY);
        	uint8_t *raw = (uint8_t*)rx_spi_buffer;

		  for(int i=0; i<DIM_TEST_PACKET; i++)
		  {
			  test_packet_1[i] = raw[(HALF_NUM_OF_DATA*2 - DIM_TEST_PACKET) + i];
		  }

		  for(int i=0; i<DIM_TEST_PACKET; i++)
		  {
			  test_packet_2[i] = raw[(NUM_OF_DATA*2 - DIM_TEST_PACKET) + i];
		  }

		  uint8_t match = (memcmp(test_packet_1, test_packet_2, DIM_TEST_PACKET) == 0);

		  uint8_t match_ref = (memcmp(test_packet_1, ref_packet, DIM_TEST_PACKET) == 0);


		  if(match && match_ref)
		  {
			  HAL_UART_Transmit(&huart1, (uint8_t*)"OK\r\n", 4, HAL_MAX_DELAY);
			  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
		  }
		  HAL_TIM_Base_Stop_IT(&htim2);
		  HAL_NVIC_DisableIRQ(TIM2_IRQn);
		  break;
        }
        default: break;
    }
    delay_counter++;
}





void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == GPIO_PIN_12 && acq_enabled)
    {
        if (HAL_SPI_GetState(&hspi1) == HAL_SPI_STATE_READY)
        {
            HAL_SPI_Receive_DMA(&hspi1, (uint8_t*)spi_rx_buffer, NUM_CHANNELS);
        }
    }
}


void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if(hspi->Instance == SPI1 && uart_tx_done && acq_enabled)
    {

      memcpy(spi_rx_snapshot, spi_rx_buffer, sizeof(spi_rx_snapshot)); //si lavora su una copia dei dati 

      //Preparo i dati per il force reconstructor
      for(uint8_t i = 0; i < force_reconstructor.n_sensors; i++)
      {
          force_input[i] = (float)spi_rx_snapshot[i];
      }

      integral_step(&force_reconstructor, force_input); //ricostruzione della forza

      // Header
      pkt.header[0] = 0x3C;
      pkt.header[1] = 0x3E;
      pkt.header[2] = 0x00;

      // Timestamp (µs)
      pkt.timestamp_us = __HAL_TIM_GET_COUNTER(&htim5);

      uint8_t idx = 0;
      
      for (int i = 0; i < NUM_CHANNELS; i++)
      {
        if (channel_mask[i])
        {
          pkt.ch[idx++] = __builtin_bswap16(spi_rx_buffer[i]);
        }
      }
      pkt.n_channels = idx;

      for(uint8_t i = 0; i < force_reconstructor.n_sensors; i++)
      {
        float force = force_reconstructor.current_output[i];

        uint16_t force_uint16 = (uint16_t)round(force * 65535.0f);

        pkt.force[i] = __builtin_bswap16(force_uint16);
      }

      //Assemblo il pacchetto e lo invio via UART
      uint16_t header_and_info_size = sizeof(pkt.header) + sizeof(pkt.n_channels) + sizeof(pkt.timestamp_us);
      uint16_t channels_size = idx * sizeof(int16_t);
      uint16_t force_size = force_reconstructor.n_sensors * sizeof(uint16_t);
      uint16_t offset = 0;

      // Copia Header, n_channels e Timestamp
      memcpy(&tx_buffer[offset], &pkt, header_and_info_size);
      offset += header_and_info_size;

      // Copia solo i canali attivi (idx elementi)
      memcpy(&tx_buffer[offset], pkt.ch, channels_size);
      offset += channels_size;

      // Copia gli 8 valori di forza subito dopo i canali attivi
      memcpy(&tx_buffer[offset], pkt.force, force_size);
      offset += force_size;
      
      uart_tx_done = 0;
      HAL_UART_Transmit_DMA(&huart1, tx_buffer, offset);
          
    }
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if(huart->Instance == USART1)
    {
        uart_tx_done = 1; // pronto per prossima trasmissione
    }
}


void parse_start_command(char *cmd)
{
    // reset
    memset(channel_mask, 0, sizeof(channel_mask));
    num_active_channels = 0;
    reset_all(&force_reconstructor); //azzera integral, max_integral, FSM e rilancia calibrazione soglie

    char *p = strchr(cmd, ',');
    if (!p)
    {
        // solo "start"
        enable_all_channels();
        return;
    }

    p++; // dopo "start,"

    while (*p && num_active_channels < MAX_SELECTED_CHANNELS)
    {
        int a = 0, b = 0;

        if (sscanf(p, "%d-%d", &a, &b) == 2)
        {
            if (a < 1) a = 1;
            if (b > NUM_CHANNELS) b = NUM_CHANNELS;

            for (int i = a; i <= b; i++)
            {
                if (!channel_mask[i-1])
                {
                    channel_mask[i-1] = 1;
                    num_active_channels++;
                }
            }
        }
        else if (sscanf(p, "%d", &a) == 1)
        {
            if (a >= 1 && a <= NUM_CHANNELS && !channel_mask[a-1])
            {
                channel_mask[a-1] = 1;
                num_active_channels++;
            }
        }

        // vai al prossimo token
        p = strchr(p, ',');
        if (!p) break;
        p++;
    }
}

void parse_configuration_command(char *cmd)
{
    char parameter[32];

    char *p = strchr(cmd, ','); //cerca la prima virgola
    if (!p) return;

    p++;  // dopo "set,"

    char *value_str = strchr(p, ','); //cerca la seconda virgola
    if (!value_str) return;

    size_t param_len = value_str - p; // lunghezza del parametro serve a evitare parametri vuoti o troppo lunghi (parameter[32])

    if (param_len == 0 || param_len >= sizeof(parameter))
        return;

    strncpy(parameter, p, param_len); // copia il parametro in un buffer locale
    parameter[param_len] = '\0'; //

    value_str++;   // punta al valore


    // --- PARAMETRI INTERI --- 
    if (strcmp(parameter, "nw") == 0)
    {
        uint16_t v = (uint16_t)atoi(value_str);

        if (v < 1)
            v = 1;

        force_reconstructor.NW = v;
    }

    else if (strcmp(parameter, "thr") == 0)
    {
        uint16_t v = (uint16_t)atoi(value_str);

        if (v < 10)
            v = 10;

        if (v > THR_SAMPLES)
            v = THR_SAMPLES;

        force_reconstructor.Thr_samples = v;

        if (force_reconstructor.nSamples_adaptive_offset > v)
        {
            force_reconstructor.nSamples_adaptive_offset = v;
        }
    }

    else if (strcmp(parameter, "wup") == 0)
    {
        uint16_t v = (uint16_t)atoi(value_str);

        force_reconstructor.warmup_samples = v;
    }

    else if (strcmp(parameter, "pc") == 0)
    {
        uint8_t v = (uint8_t)atoi(value_str);

        if (v < 1)
            v = 1;

        force_reconstructor.press_confirm = v;
    }

    else if (strcmp(parameter, "ao") == 0)
    {
        uint16_t v = (uint16_t)atoi(value_str);

        if (v < 1)
            v = 1;

        if (v > OFFSET_SAMPLES)
            v = OFFSET_SAMPLES;

        if (v > force_reconstructor.Thr_samples)
            v = force_reconstructor.Thr_samples;

        force_reconstructor.nSamples_adaptive_offset = v;
    }

    else if (strcmp(parameter, "sar") == 0)
    {
        uint16_t v = (uint16_t)atoi(value_str);

        force_reconstructor.samples_after_release = v;
    }

    else if (strcmp(parameter, "mp") == 0)
    {
        uint8_t v = (uint8_t)atoi(value_str);

        if (v < 1)
            v = 1;

        if (v > force_reconstructor.n_sensors)
            v = force_reconstructor.n_sensors;

        force_reconstructor.min_press_sensors = v;
    }

    else if (strcmp(parameter, "ns") == 0)
    {
        uint8_t v = (uint8_t)atoi(value_str);

        if (v < 1)
            v = 1;

        if (v > N_SENSORS)
            v = N_SENSORS;

        force_reconstructor.n_sensors = v;

        if (force_reconstructor.min_press_sensors > v)
        {
            force_reconstructor.min_press_sensors = v;
        }
    }

    // --- PARAMETRI FLOAT ---
    //separo parte intera e decimale
    else if (strcmp(parameter, "sig") == 0)
    {
        float v = 0.0f;

        int integer_part = 0;
        int decimal_part = 0;
        int decimal_digits = 0;
        int sign = 1;

        if (*value_str == '-')
        {
            sign = -1;
            value_str++;
        }

        while (*value_str >= '0' && *value_str <= '9')
        {
            integer_part = integer_part * 10 + (*value_str - '0');
            value_str++;
        }

        if (*value_str == '.')
        {
            value_str++;

            while (*value_str >= '0' && *value_str <= '9')
            {
                decimal_part = decimal_part * 10 + (*value_str - '0');
                decimal_digits++;
                value_str++;
            }
        }

        v = (float)integer_part;

        if (decimal_digits > 0)
        {
            float divisor = 1.0f;

            for (int i = 0; i < decimal_digits; i++)
                divisor *= 10.0f;

            v += (float)decimal_part / divisor;
        }

        v *= (float)sign;

        if (v < 0.0f)
            v = 0.0f;

        force_reconstructor.press_sigma = v;
    }

    else if (strcmp(parameter, "a") == 0)
    {
        float v = 0.0f;

        int integer_part = 0;
        int decimal_part = 0;
        int decimal_digits = 0;

        while (*value_str >= '0' && *value_str <= '9')
        {
            integer_part = integer_part * 10 + (*value_str - '0');
            value_str++;
        }

        if (*value_str == '.')
        {
            value_str++;

            while (*value_str >= '0' && *value_str <= '9')
            {
                decimal_part = decimal_part * 10 + (*value_str - '0');
                decimal_digits++;
                value_str++;
            }
        }

        v = (float)integer_part;

        if (decimal_digits > 0)
        {
            float divisor = 1.0f;

            for (int i = 0; i < decimal_digits; i++)
                divisor *= 10.0f;

            v += (float)decimal_part / divisor;
        }

        if (v < 0.0f)
            v = 0.0f;

        if (v > 1.0f)
            v = 1.0f;

        force_reconstructor.alpha = v;
    }

    else if (strcmp(parameter, "s2n") == 0)
    {
        float v = 0.0f;

        int integer_part = 0;
        int decimal_part = 0;
        int decimal_digits = 0;

        while (*value_str >= '0' && *value_str <= '9')
        {
            integer_part = integer_part * 10 + (*value_str - '0');
            value_str++;
        }

        if (*value_str == '.')
        {
            value_str++;

            while (*value_str >= '0' && *value_str <= '9')
            {
                decimal_part = decimal_part * 10 + (*value_str - '0');
                decimal_digits++;
                value_str++;
            }
        }

        v = (float)integer_part;

        if (decimal_digits > 0)
        {
            float divisor = 1.0f;

            for (int i = 0; i < decimal_digits; i++)
                divisor *= 10.0f;

            v += (float)decimal_part / divisor;
        }

        if (v < 0.0f)
            v = 0.0f;

        force_reconstructor.signal2noise_ratio = v;
    }

    else if (strcmp(parameter, "rr") == 0)
    {
        float v = 0.0f;

        int integer_part = 0;
        int decimal_part = 0;
        int decimal_digits = 0;

        while (*value_str >= '0' && *value_str <= '9')
        {
            integer_part = integer_part * 10 + (*value_str - '0');
            value_str++;
        }

        if (*value_str == '.')
        {
            value_str++;

            while (*value_str >= '0' && *value_str <= '9')
            {
                decimal_part = decimal_part * 10 + (*value_str - '0');
                decimal_digits++;
                value_str++;
            }
        }

        v = (float)integer_part;

        if (decimal_digits > 0)
        {
            float divisor = 1.0f;

            for (int i = 0; i < decimal_digits; i++)
                divisor *= 10.0f;

            v += (float)decimal_part / divisor;
        }

        if (v < 0.0f)
            v = 0.0f;

        if (v > 1.0f)
            v = 1.0f;

        force_reconstructor.release_ratio = v;
    }
}

void parse_resetDefault_command(char *cmd)
{
    //Reset dei parametri ai valori di default e rilancio della calibrazione delle soglie, integral, max_integral e FSM vengono azzerati
    if (strncmp(cmd, "rd", 2) == 0)
    {
      ForceReconstructor_Init(&force_reconstructor); 
    }
}

void parse_reset_command(char *cmd)
{
    //Reset integral, max_integral, FSM e rilancio della calibrazione delle soglie, i parametri configurati rimangono invariati
    if (strncmp(cmd, "r", 1) == 0)
    {
      reset_all(&force_reconstructor);
    }
}

static void enable_all_channels(void)
{
    for (int i = 0; i < NUM_CHANNELS; i++)
        channel_mask[i] = 1;

    num_active_channels = NUM_CHANNELS;
}

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  //MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();



}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
