// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_UART_APB_H
#define ARA_VHK158_UART_APB_H

#include <stdint.h>

#define UART_BASE 0xC0000000UL

#include "platform.h"
#ifndef SOC_CLK_HZ
#define SOC_CLK_HZ ARA_SOC_CLK_HZ
#endif

#define UART_DIVISOR ((SOC_CLK_HZ + 8u * 115200u) / (16u * 115200u))

#define UART_THR  (*(volatile uint32_t *)(UART_BASE + 0x00))
#define UART_DLL  (*(volatile uint32_t *)(UART_BASE + 0x00))
#define UART_DLM  (*(volatile uint32_t *)(UART_BASE + 0x04))
#define UART_LCR  (*(volatile uint32_t *)(UART_BASE + 0x0C))
#define UART_LSR  (*(volatile uint32_t *)(UART_BASE + 0x14))

#define UART_LCR_DLAB     0x80
#define UART_LCR_8N1      0x03
#define UART_LSR_THRE     0x20

static inline void uart_init(uint32_t divisor) {
    UART_LCR = UART_LCR_DLAB;
    UART_DLL = divisor & 0xff;
    UART_DLM = (divisor >> 8) & 0xff;
    UART_LCR = UART_LCR_8N1;
}

static inline void uart_putc(char c) {
    while (!(UART_LSR & UART_LSR_THRE)) {  }
    UART_THR = (uint32_t)(unsigned char)c;
}

#endif
