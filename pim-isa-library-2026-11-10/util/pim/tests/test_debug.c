/*
 * L4: the PIM_DEBUG misuse checks (LQ9 (c)) abort with a message; each
 * misuse runs in a child process.
 */

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pim_internal.h"
#include "pim_test.h"

#ifndef PIM_DEBUG
#error "test_debug is built with PIM_DEBUG"
#endif

static pim_buf_t *buf;

static void
misuse(int which)
{
    pim_coord_t c = {0, 0, 0, 0, 0, 0, 0};
    uint8_t data[8] = {0};
    uint8_t *p = (uint8_t *)pim_buf_addr(buf, &c);
    switch (which) {
        case 0: /* none: allocated row, inside one piece */
            pim_ab_mac(p, PIM_LAYOUT_COL_BYTES);
            pim_sb_mac(p, PIM_DESC(1));
            pim_wr_gb_mem(pim_gb_addr(0, 0, 0), data, 8);
            pim_rd_reg_mem(pim_acc_addr(&c), data, 8);
            break;
        case 1: /* a row outside every buffer */
            pim_ab_mac(p + ((uint64_t)8 << PIM_LAYOUT_ROW_SHIFT),
                       PIM_LAYOUT_COL_BYTES);
            break;
        case 2: /* a range across a contiguous piece */
            pim_ab_mac(p + PIM_LAYOUT_CONTIG_BYTES - 8, 16);
            break;
        case 3: /* a scalar address not 8-byte aligned */
            pim_sb_mac(p + 4, PIM_DESC(1));
            break;
        case 4: /* reserved descriptor bits */
            pim_sb_mac(p, 1ULL << 16);
            break;
        case 5: /* a register access outside the window */
            pim_rd_reg_mem(p, data, 8);
            break;
        case 6: /* register index NUM_REGS */
            pim_reg_addr(&c, PIM_LAYOUT_NUM_REGS);
            break;
        case 7: /* outside the PIM memory */
            pim_ab_mac(data, 8);
            break;
    }
}

int
main(void)
{
    if (pim_init(NULL) != PIM_OK || pim_buf_alloc(&buf, 2) != PIM_OK) {
        printf("FAIL: init\n");
        return 1;
    }
    for (int which = 0; which <= 7; which++) {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            misuse(which);
            _exit(0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        const int aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
        CHECK(which == 0 ? WIFEXITED(status) && WEXITSTATUS(status) == 0
                         : aborted,
              "misuse %d: status %#x", which, status);
    }
    pim_finalize();
    return pim_test_summary("test_debug");
}
