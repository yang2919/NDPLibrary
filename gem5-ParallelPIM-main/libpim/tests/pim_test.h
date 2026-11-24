/*
 * A minimal test harness for the PIM library's host tests.
 */

#ifndef __PIM_TEST_H__
#define __PIM_TEST_H__

#include <stdio.h>

static long pim_test_checks;
static long pim_test_failures;

/* CHECK(condition, format, ...): count, and report the first failures. */
#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        pim_test_checks++;                                                    \
        if (!(cond)) {                                                        \
            if (pim_test_failures++ < 20) {                                   \
                fprintf(stderr, "%s:%d: CHECK(%s) failed: ", __FILE__,        \
                        __LINE__, #cond);                                     \
                fprintf(stderr, __VA_ARGS__);                                 \
                fputc('\n', stderr);                                          \
            }                                                                 \
        }                                                                     \
    } while (0)

static inline int
pim_test_summary(const char *name)
{
    printf("%s: %ld checks, %ld failures\n", name, pim_test_checks,
           pim_test_failures);
    return pim_test_failures != 0;
}

#endif // __PIM_TEST_H__
