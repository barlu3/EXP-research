/*
 * Round CORE-MATH lookup tables to 16-bit MPFR precision.
 *
 * Each original float value (binary32, 24-bit significand) is loaded into a
 * 24-bit MPFR number — an exact conversion — then rounded to 16-bit MPFR
 * precision with MPFR_RNDN.  The function whose tables are rounded is chosen
 * at compile time (exactly one):
 *   (default)     exp — T1/T2          -> round-16bitp.txt
 *   -DROUND_SIN   sin — S1/C1/S2/C2/S3/C3 -> sin/round-16bitp-sin.txt
 *   -DROUND_LOG   log — T1/T2/T3        -> log/round-16bitp-log.txt
 *
 * Compile (from cross-eval/):
 *   gcc -O2 -std=c11             round-16bitp.c -lmpfr -lgmp -lm -o round-16bitp
 *   gcc -O2 -std=c11 -DROUND_SIN round-16bitp.c -lmpfr -lgmp -lm -o round-sin
 *   gcc -O2 -std=c11 -DROUND_LOG round-16bitp.c -lmpfr -lgmp -lm -o round-log
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpfr.h>

/* Select exactly one function's tables.  Default to exp when none is given. */
#if !defined(ROUND_SIN) && !defined(ROUND_LOG) && !defined(ROUND_EXP)
#define ROUND_EXP
#endif

/* ── exp(x) tables (from inria-exp16bitp.c) ─────────────────────────── */
#if defined(ROUND_EXP)
static const float T1[] = {
 0x1.00802p+0, 0x1.008824p+0, 0x1.009028p+0, 0x1.00982ep+0, 0x1.00a032p+0,
 0x1.00a838p+0, 0x1.00b03cp+0, 0x1.00b842p+0, 0x1.00c048p+0, 0x1.00c84ep+0,
 0x1.00d054p+0, 0x1.00d85cp+0, 0x1.00e062p+0, 0x1.00e86ap+0, 0x1.00f07p+0,
 0x1.00f878p+0, 0x1.01008p+0, 0x1.01109p+0, 0x1.0120a2p+0, 0x1.0130b4p+0,
 0x1.0140c8p+0, 0x1.0150dcp+0, 0x1.0160f2p+0, 0x1.017108p+0, 0x1.01812p+0,
 0x1.01913ap+0, 0x1.01a152p+0, 0x1.01b16ep+0, 0x1.01c188p+0, 0x1.01d1a6p+0,
 0x1.01e1c4p+0, 0x1.01f1e2p+0, 0x1.020202p+0, 0x1.022244p+0, 0x1.02428ap+0,
 0x1.0262d4p+0, 0x1.028322p+0, 0x1.02a376p+0, 0x1.02c3ccp+0, 0x1.02e426p+0,
 0x1.030484p+0, 0x1.0324e8p+0, 0x1.03454ep+0, 0x1.0365b8p+0, 0x1.038628p+0,
 0x1.03a69ap+0, 0x1.03c71p+0, 0x1.03e78cp+0, 0x1.04080ap+0, 0x1.044914p+0,
 0x1.048a3p+0, 0x1.04cb5ap+0, 0x1.050c94p+0, 0x1.054dep+0, 0x1.058f3cp+0,
 0x1.05d0a8p+0, 0x1.061224p+0, 0x1.0653bp+0, 0x1.06954ep+0, 0x1.06d6fcp+0,
 0x1.0718bap+0, 0x1.075a88p+0, 0x1.079c66p+0, 0x1.07de56p+0, 0x1.082056p+0,
 0x1.08a488p+0, 0x1.0928fap+0, 0x1.09adbp+0, 0x1.0a32a8p+0, 0x1.0ab7e2p+0,
 0x1.0b3d6p+0, 0x1.0bc32p+0, 0x1.0c4924p+0, 0x1.0ccf6ap+0, 0x1.0d55f2p+0,
 0x1.0ddccp+0, 0x1.0e63dp+0, 0x1.0eeb24p+0, 0x1.0f72bap+0, 0x1.0ffa96p+0,
 0x1.1082b6p+0, 0x1.1193cp+0, 0x1.12a5dep+0, 0x1.13b90cp+0, 0x1.14cd5p+0,
 0x1.15e2a8p+0, 0x1.16f916p+0, 0x1.18109ap+0, 0x1.192938p+0, 0x1.1a42eep+0,
 0x1.1b5dbep+0, 0x1.1c79a8p+0, 0x1.1d96bp+0, 0x1.1eb4d6p+0, 0x1.1fd41ap+0,
 0x1.20f48p+0, 0x1.221604p+0, 0x1.245c76p+0, 0x1.26a77ap+0, 0x1.28f718p+0,
 0x1.2b4b58p+0, 0x1.2da448p+0, 0x1.3001ecp+0, 0x1.326452p+0, 0x1.34cb82p+0,
 0x1.373784p+0, 0x1.39a862p+0, 0x1.3c1e28p+0, 0x1.3e98dep+0, 0x1.41189p+0,
 0x1.439d44p+0, 0x1.462708p+0, 0x1.48b5e4p+0, 0x1.4de30ep+0, 0x1.532518p+0,
 0x1.587c54p+0, 0x1.5de918p+0, 0x1.636bbap+0, 0x1.690492p+0, 0x1.6eb3fcp+0,
 0x1.747a52p+0, 0x1.7a57eep+0, 0x1.804d3p+0, 0x1.865a78p+0, 0x1.8c8024p+0,
 0x1.92be9ap+0, 0x1.99163ap+0, 0x1.9f876ep+0, 0x1.a61298p+0, 0x1.b3787ep+0,
 0x1.c14b44p+0, 0x1.cf8e5ep+0, 0x1.de455ep+0, 0x1.ed73f2p+0, 0x1.fd1de6p+0,
 0x1.06a392p+1, 0x1.0ef9dcp+1, 0x1.1793e4p+1, 0x1.2073d4p+1, 0x1.299be2p+1,
 0x1.330e58p+1, 0x1.3ccd94p+1, 0x1.46dc04p+1, 0x1.513c2ep+1, 0x1.5bf0a8p+1,
 0x1.72615ap+1, 0x1.8a448cp+1, 0x1.a3b222p+1, 0x1.bec38ep+1, 0x1.db93e4p+1,
 0x1.fa3ff4p+1, 0x1.0d7338p+2, 0x1.1ed3fep+2, 0x1.3153b2p+2, 0x1.4504d2p+2,
 0x1.59fb12p+2, 0x1.704b6ap+2, 0x1.880c2ap+2, 0x1.a1551ap+2, 0x1.bc3f8p+2,
 0x1.d8e64cp+2, 0x1.0beec6p+3, 0x1.2f9b88p+3, 0x1.58084cp+3, 0x1.85d6fep+3,
 0x1.b9bf12p+3, 0x1.f4907p+3, 0x1.1b9b56p+4, 0x1.415e5cp+4, 0x1.6c2888p+4,
 0x1.9ca53cp+4, 0x1.d396aap+4, 0x1.08ec72p+5, 0x1.2c32a2p+5, 0x1.542b2ep+5,
 0x1.81762ap+5, 0x1.b4c902p+5, 0x1.186bf2p+6, 0x1.68118ap+6, 0x1.ce564ep+6,
 0x1.28d38ap+7, 0x1.7d21eep+7, 0x1.e96244p+7, 0x1.3a30dp+8, 0x1.936dc6p+8,
 0x1.0301a4p+9, 0x1.4c9222p+9, 0x1.ab0786p+9, 0x1.122886p+10, 0x1.6006b6p+10,
 0x1.c402b6p+10, 0x1.223252p+11, 0x1.749ea8p+11, 0x1.332c4ep+12,
 0x1.fa7158p+12, 0x1.a17ddp+13, 0x1.5829dcp+14, 0x1.1bb702p+15,
 0x1.d3c448p+15, 0x1.819bc6p+16, 0x1.3de166p+17, 0x1.060c52p+18,
 0x1.b00b5ap+18, 0x1.64290cp+19, 0x1.259ac4p+20, 0x1.e41274p+20,
 0x1.8f0ccap+21, 0x1.48f60ap+22, 0x1.0f2ebep+23, 0x1.709348p+24,
 0x1.f4f22p+25, 0x1.546d9p+27, 0x1.ceb088p+28, 0x1.3a6e2p+30, 0x1.ab5adcp+31,
 0x1.226af4p+33, 0x1.8ab7fcp+34, 0x1.0c3d3ap+36, 0x1.6c9326p+37,
 0x1.ef823p+38, 0x1.50bba4p+40, 0x1.c9aae4p+41, 0x1.37047p+43, 0x1.a6b766p+44,
 0x1.1f43fcp+46, 0x1.0953e2p+49, 0x1.ea215ap+51, 0x1.c4b334p+54,
 0x1.a220d4p+57, 0x1.823256p+60, 0x1.64b41cp+63, 0x1.49767cp+66,
 0x1.304d6ap+69, 0x1.19103ep+72, 0x1.039966p+75, 0x1.df8c5ap+77,
 0x1.baed16p+80, 0x1.9919cap+83, 0x1.79dbcap+86, 0x1.5d0094p+89,
 0x1.425982p+92, 0x1.12fec8p+98, 0x1.d531d8p+103, 0x1.9044a8p+109,
 0x1.55779cp+115, 0x1.234deap+121, 0x1.f1056ep+126, 0x1.fffffep+127,
 0x1.fffffep+127, 0x1.fffffep+127, 0x1.fffffep+127, 0x1.fffffep+127,
 0x1.fffffep+127, 0x1.fffffep+127, 0x1.fffffep+127, 0x1.fffffep+127,
 0x1.ff004p-1, 0x1.fef048p-1, 0x1.fee05p-1, 0x1.fed05ap-1, 0x1.fec064p-1,
 0x1.feb06ep-1, 0x1.fea078p-1, 0x1.fe9084p-1, 0x1.fe809p-1, 0x1.fe709cp-1,
 0x1.fe60a8p-1, 0x1.fe50b6p-1, 0x1.fe40c4p-1, 0x1.fe30d2p-1, 0x1.fe20ep-1,
 0x1.fe10fp-1, 0x1.fe01p-1, 0x1.fde12p-1, 0x1.fdc144p-1, 0x1.fda168p-1,
 0x1.fd819p-1, 0x1.fd61b8p-1, 0x1.fd41e4p-1, 0x1.fd221p-1, 0x1.fd023ep-1,
 0x1.fce27p-1, 0x1.fcc2a2p-1, 0x1.fca2d8p-1, 0x1.fc830ep-1, 0x1.fc6348p-1,
 0x1.fc4382p-1, 0x1.fc23bep-1, 0x1.fc03fep-1, 0x1.fbc48p-1, 0x1.fb850cp-1,
 0x1.fb45ap-1, 0x1.fb063ap-1, 0x1.fac6dep-1, 0x1.fa878ap-1, 0x1.fa483cp-1,
 0x1.fa08f8p-1, 0x1.f9c9bap-1, 0x1.f98a84p-1, 0x1.f94b58p-1, 0x1.f90c32p-1,
 0x1.f8cd14p-1, 0x1.f88dfep-1, 0x1.f84efp-1, 0x1.f80feap-1, 0x1.f791f6p-1,
 0x1.f71422p-1, 0x1.f6966cp-1, 0x1.f618d6p-1, 0x1.f59b6p-1, 0x1.f51e08p-1,
 0x1.f4a0dp-1, 0x1.f423b8p-1, 0x1.f3a6cp-1, 0x1.f329e6p-1, 0x1.f2ad2ap-1,
 0x1.f2308ep-1, 0x1.f1b412p-1, 0x1.f137b4p-1, 0x1.f0bb76p-1, 0x1.f03f56p-1,
 0x1.ef4774p-1, 0x1.ee501p-1, 0x1.ed5924p-1, 0x1.ec62b6p-1, 0x1.eb6cc2p-1,
 0x1.ea774ap-1, 0x1.e9824ap-1, 0x1.e88dc6p-1, 0x1.e799bcp-1, 0x1.e6a62cp-1,
 0x1.e5b316p-1, 0x1.e4c07ap-1, 0x1.e3ce56p-1, 0x1.e2dcacp-1, 0x1.e1eb7ap-1,
 0x1.e0facp-1, 0x1.df1ab6p-1, 0x1.dd3c8ap-1, 0x1.db603cp-1, 0x1.d985c8p-1,
 0x1.d7ad3p-1, 0x1.d5d66ep-1, 0x1.d40182p-1, 0x1.d22e6ap-1, 0x1.d05d24p-1,
 0x1.ce8dbp-1, 0x1.ccc008p-1, 0x1.caf42ep-1, 0x1.c92a2p-1, 0x1.c761dap-1,
 0x1.c59b5cp-1, 0x1.c3d6a2p-1, 0x1.c0527ap-1, 0x1.bcd554p-1, 0x1.b95f2p-1,
 0x1.b5efd2p-1, 0x1.b2875cp-1, 0x1.af25bp-1, 0x1.abcac2p-1, 0x1.a87682p-1,
 0x1.a528e2p-1, 0x1.a1e1dap-1, 0x1.9ea158p-1, 0x1.9b675p-1, 0x1.9833b6p-1,
 0x1.95067cp-1, 0x1.91df98p-1, 0x1.8ebefap-1, 0x1.889064p-1, 0x1.827a56p-1,
 0x1.7c7c7p-1, 0x1.769652p-1, 0x1.70c79ep-1, 0x1.6b0ff8p-1, 0x1.656fp-1,
 0x1.5fe462p-1, 0x1.5a6fcp-1, 0x1.5510c6p-1, 0x1.4fc71ep-1, 0x1.4a9272p-1,
 0x1.45726ep-1, 0x1.4066c2p-1, 0x1.3b6f1ep-1, 0x1.368b3p-1, 0x1.2cfd4p-1,
 0x1.23ba94p-1, 0x1.1ac0d6p-1, 0x1.120dcap-1, 0x1.099f42p-1, 0x1.017324p-1,
 0x1.f30ec8p-2, 0x1.e3b40ep-2, 0x1.d4d244p-2, 0x1.c665b2p-2, 0x1.b86abap-2,
 0x1.aaddep-2, 0x1.9dbbcp-2, 0x1.91011p-2, 0x1.84aaa4p-2, 0x1.78b564p-2,
 0x1.61e28ap-2, 0x1.4c71b2p-2, 0x1.384d68p-2, 0x1.256184p-2, 0x1.139b1ap-2,
 0x1.02e862p-2, 0x1.e6715p-3, 0x1.c8f878p-3, 0x1.ad48bcp-3, 0x1.93466ep-3,
 0x1.7ad788p-3, 0x1.63e398p-3, 0x1.4e53aap-3, 0x1.3a122cp-3, 0x1.270adap-3,
 0x1.152aaap-3, 0x1.e9328cp-4, 0x1.afb718p-4, 0x1.7cfcc2p-4, 0x1.50385cp-4,
 0x1.28b698p-4, 0x1.05d938p-4, 0x1.ce2938p-5, 0x1.97db0cp-5, 0x1.67ee6ep-5,
 0x1.3da368p-5, 0x1.185098p-5, 0x1.eec102p-6, 0x1.b49e6ap-6, 0x1.815094p-6,
 0x1.540ap-6, 0x1.2c155cp-6, 0x1.d36912p-7, 0x1.6c0504p-7, 0x1.1b7faep-7,
 0x1.b993fep-8, 0x1.57e6cp-8, 0x1.0bd4a6p-8, 0x1.a12c66p-9, 0x1.44e52p-9,
 0x1.fa0e96p-10, 0x1.8a1e18p-10, 0x1.32f066p-10, 0x1.de16bap-11,
 0x1.7455fep-11, 0x1.21f9bap-11, 0x1.c3aa92p-12, 0x1.5fc21p-12, 0x1.aab44p-13,
 0x1.02cf22p-13, 0x1.39f38ap-14, 0x1.7cd79cp-15, 0x1.cdfc26p-16,
 0x1.183542p-16, 0x1.53e8d8p-17, 0x1.9c54c4p-18, 0x1.f42ed4p-19,
 0x1.2f6054p-19, 0x1.700398p-20, 0x1.be6c7p-21, 0x1.0ec504p-21,
 0x1.4875cap-22, 0x1.8e7138p-23, 0x1.e355bcp-24, 0x1.639e32p-25,
 0x1.05a628p-26, 0x1.81057p-28, 0x1.1b4866p-29, 0x1.a0db0ep-31,
 0x1.32b48cp-32, 0x1.c3527ep-34, 0x1.4c1078p-35, 0x1.e8a37ap-37,
 0x1.67852ap-38, 0x1.08852ap-39, 0x1.853f02p-41, 0x1.1e642cp-42,
 0x1.a56e0cp-44, 0x1.36121ep-45, 0x1.c8465p-47, 0x1.ee001ep-50,
 0x1.0b6c3ap-52, 0x1.2188aep-55, 0x1.397924p-58, 0x1.536452p-61,
 0x1.6f741ep-64, 0x1.8dd5e2p-67, 0x1.aebabap-70, 0x1.d257d6p-73,
 0x1.f8e6c2p-76, 0x1.1152eap-78, 0x1.27ec46p-81, 0x1.4063f8p-84,
 0x1.5ae192p-87, 0x1.778fe2p-90, 0x1.969d48p-93, 0x1.dca23cp-99,
 0x1.175afp-104, 0x1.4775ep-110, 0x1.7fd974p-116, 0x1.c1f2dap-122,
 0x1.07b712p-127, 0x1.352088p-133, 0x1.6a5beap-139, 0x1.a8c1f2p-145,
 0x1.f1e6b6p-151, 0x1.23d1f4p-156, 0x1.561262p-162, 0x1.90fa16p-168,
 0x1.d60684p-174, 0x1.137b6cp-179,
};

/* For 0 <= i2 < 2^7, T2[i2] stores the binary32 approximation of y2=exp(x2)
   to nearest, where x2 = 2^(h-16)*l, i2 decomposes into 2^3*h+l,
   0 <= h < 2^4, 0 <= l < 2^3, and T2[2^7+i2] stores the binary32
   approximation of y2=exp(-x2) to nearest.
   One entry was manually optimized to avoid an exception.
   Table generated by table2() from exp.sage. */
static const float T2[] = {
 0x1p+0, 0x1.0001p+0, 0x1.0002p+0, 0x1.0003p+0, 0x1.0004p+0, 0x1.0005p+0,
 0x1.0006p+0, 0x1.0007p+0, 0x1p+0, 0x1.0002p+0, 0x1.0004p+0, 0x1.0006p+0,
 0x1.0008p+0, 0x1.000ap+0, 0x1.000cp+0, 0x1.000ep+0, 0x1p+0, 0x1.0004p+0,
 0x1.0008p+0, 0x1.000cp+0, 0x1.001p+0, 0x1.0014p+0, 0x1.001802p+0,
 0x1.001c02p+0, 0x1p+0, 0x1.0008p+0, 0x1.001p+0, 0x1.001802p+0, 0x1.002002p+0,
 0x1.002804p+0, 0x1.003004p+0, 0x1.003806p+0, 0x1p+0, 0x1.001p+0,
 0x1.002002p+0, 0x1.003004p+0, 0x1.004008p+0, 0x1.00500cp+0, 0x1.006012p+0,
 0x1.007018p+0, 0x1p+0, 0x1.002002p+0, 0x1.004008p+0, 0x1.006012p+0,
 0x1.00802p+0, 0x1.00a032p+0, 0x1.00c048p+0, 0x1.00e062p+0, 0x1p+0,
 0x1.004008p+0, 0x1.00802p+0, 0x1.00c048p+0, 0x1.01008p+0, 0x1.0140c8p+0,
 0x1.01812p+0, 0x1.01c188p+0, 0x1p+0, 0x1.00802p+0, 0x1.01008p+0,
 0x1.01812p+0, 0x1.020202p+0, 0x1.028322p+0, 0x1.030484p+0, 0x1.038628p+0,
 0x1p+0, 0x1.01008p+0, 0x1.020202p+0, 0x1.030484p+0, 0x1.04080ap+0,
 0x1.050c94p+0, 0x1.061224p+0, 0x1.0718bap+0, 0x1p+0, 0x1.020202p+0,
 0x1.04080ap+0, 0x1.061224p+0, 0x1.082056p+0, 0x1.0a32a8p+0, 0x1.0c4924p+0,
 0x1.0e63dp+0, 0x1p+0, 0x1.04080ap+0, 0x1.082056p+0, 0x1.0c4924p+0,
 0x1.1082b6p+0, 0x1.14cd5p+0, 0x1.192938p+0, 0x1.1d96bp+0, 0x1p+0,
 0x1.082056p+0, 0x1.1082b6p+0, 0x1.192938p+0, 0x1.221604p+0, 0x1.2b4b58p+0,
 0x1.34cb82p+0, 0x1.3e98dep+0, 0x1p+0, 0x1.1082b6p+0, 0x1.221604p+0,
 0x1.34cb82p+0, 0x1.48b5e4p+0, 0x1.5de918p+0, 0x1.747a52p+0, 0x1.8c8024p+0,
 0x1p+0, 0x1.221604p+0, 0x1.48b5e4p+0, 0x1.747a52p+0, 0x1.a61298p+0,
 0x1.de455ep+0, 0x1.0ef9dcp+1, 0x1.330e58p+1, 0x1p+0, 0x1.48b5e4p+0,
 0x1.a61298p+0, 0x1.0ef9dcp+1, 0x1.5bf0a8p+1, 0x1.bec38ep+1, 0x1.1ed3fep+2,
 0x1.704b6ap+2, 0x1.000002p+0, 0x1.a61298p+0, 0x1.5bf0a8p+1, 0x1.1ed3fep+2,
 0x1.d8e64cp+2, 0x1.85d6fep+3, 0x1.415e5cp+4, 0x1.08ec72p+5, 0x1p+0,
 0x1.fffep-1, 0x1.fffcp-1, 0x1.fffap-1, 0x1.fff8p-1, 0x1.fff6p-1, 0x1.fff4p-1,
 0x1.fff2p-1, 0x1p+0, 0x1.fffcp-1, 0x1.fff8p-1, 0x1.fff4p-1, 0x1.fffp-1,
 0x1.ffecp-1, 0x1.ffe8p-1, 0x1.ffe4p-1, 0x1p+0, 0x1.fff8p-1, 0x1.fffp-1,
 0x1.ffe8p-1, 0x1.ffep-1, 0x1.ffd802p-1, 0x1.ffd002p-1, 0x1.ffc804p-1, 0x1p+0,
 0x1.fffp-1, 0x1.ffep-1, 0x1.ffd002p-1, 0x1.ffc004p-1, 0x1.ffb006p-1,
 0x1.ffa008p-1, 0x1.ff900cp-1, 0x1p+0, 0x1.ffep-1, 0x1.ffc004p-1,
 0x1.ffa008p-1, 0x1.ff801p-1, 0x1.ff6018p-1, 0x1.ff4024p-1, 0x1.ff203p-1,
 0x1p+0, 0x1.ffc004p-1, 0x1.ff801p-1, 0x1.ff4024p-1, 0x1.ff004p-1,
 0x1.fec064p-1, 0x1.fe809p-1, 0x1.fe40c4p-1, 0x1p+0, 0x1.ff801p-1,
 0x1.ff004p-1, 0x1.fe809p-1, 0x1.fe01p-1, 0x1.fd819p-1, 0x1.fd023ep-1,
 0x1.fc830ep-1, 0x1p+0, 0x1.ff004p-1, 0x1.fe01p-1, 0x1.fd023ep-1,
 0x1.fc03fep-1, 0x1.fb063ap-1, 0x1.fa08f8p-1, 0x1.f90c32p-1, 0x1p+0,
 0x1.fe01p-1, 0x1.fc03fep-1, 0x1.fa08f8p-1, 0x1.f80feap-1, 0x1.f618d6p-1,
 0x1.f423b8p-1, 0x1.f2308ep-1, 0x1p+0, 0x1.fc03fep-1, 0x1.f80feap-1,
 0x1.f423b8p-1, 0x1.f03f56p-1, 0x1.ec62b6p-1, 0x1.e88dc6p-1, 0x1.e4c07ap-1,
 0x1p+0, 0x1.f80feap-1, 0x1.f03f56p-1, 0x1.e88dc6p-1, 0x1.e0facp-1,
 0x1.d985c8p-1, 0x1.d22e6ap-1, 0x1.caf42ep-1, 0x1p+0, 0x1.f03f56p-1,
 0x1.e0facp-1, 0x1.d22e6ap-1, 0x1.c3d6a2p-1, 0x1.b5efd2p-1, 0x1.a87682p-1,
 0x1.9b675p-1, 0x1p+0, 0x1.e0facp-1, 0x1.c3d6a2p-1, 0x1.a87682p-1,
 0x1.8ebefap-1, 0x1.769652p-1, 0x1.5fe462p-1, 0x1.4a9272p-1, 0x1p+0,
 0x1.c3d6a2p-1, 0x1.8ebefap-1, 0x1.5fe462p-1, 0x1.368b3p-1, 0x1.120dcap-1,
 0x1.e3b40ep-2, 0x1.aaddep-2, 0x1p+0, 0x1.8ebefap-1, 0x1.368b3p-1,
 0x1.e3b40ep-2, 0x1.78b564p-2, 0x1.256184p-2, 0x1.c8f878p-3, 0x1.63e398p-3,
 0x1p+0, 0x1.368b3p-1, 0x1.78b564p-2, 0x1.c8f878p-3, 0x1.152aaap-3,
 0x1.50385cp-4, 0x1.97db0cp-5, 0x1.eec102p-6,
};
#endif /* ROUND_EXP */

/* ── sin(x) tables (from inria-sinbf16.c) ───────────────────────────── */
#if defined(ROUND_SIN)
/* For 0 <= i1 < 2^8, S1[i1] stores the binary32 approximation of sin(x1)
   to nearest, where the bfloat16 encoding of x1 is 123*2^7+i1*2^3,
   Table generated by tableS1() from sin.sage. */
static const float S1[] = {
 0x1.ffaaaep-5, 0x1.0fccd6p-4, 0x1.1fc344p-4, 0x1.2fb892p-4, 0x1.3facb2p-4,
 0x1.4f9f9p-4, 0x1.5f912p-4, 0x1.6f815p-4, 0x1.7f701p-4, 0x1.8f5d52p-4,
 0x1.9f4902p-4, 0x1.af3316p-4, 0x1.bf1b78p-4, 0x1.cf021cp-4, 0x1.dee6f2p-4,
 0x1.eec9e8p-4, 0x1.feaaeep-4, 0x1.0f3378p-3, 0x1.1f0d3ep-3, 0x1.2ee286p-3,
 0x1.3eb312p-3, 0x1.4e7ea4p-3, 0x1.5e44fcp-3, 0x1.6e05dcp-3, 0x1.7dc102p-3,
 0x1.8d7632p-3, 0x1.9d252ep-3, 0x1.accdb2p-3, 0x1.bc6f84p-3, 0x1.cc0a66p-3,
 0x1.db9e16p-3, 0x1.eb2a58p-3, 0x1.faaeeep-3, 0x1.0cd00cp-2, 0x1.1c37d6p-2,
 0x1.2b8ddcp-2, 0x1.3ad12ap-2, 0x1.4a00cap-2, 0x1.591bcap-2, 0x1.682138p-2,
 0x1.771026p-2, 0x1.85e7a2p-2, 0x1.94a6bep-2, 0x1.a34c92p-2, 0x1.b1d83p-2,
 0x1.c048b2p-2, 0x1.ce9d2ep-2, 0x1.dcd4c2p-2, 0x1.eaee88p-2, 0x1.036294p-1,
 0x1.110d0cp-1, 0x1.1e7344p-1, 0x1.2b91dep-1, 0x1.386598p-1, 0x1.44eb38p-1,
 0x1.511fap-1, 0x1.5cffc2p-1, 0x1.6888a4p-1, 0x1.73b768p-1, 0x1.7e894p-1,
 0x1.88fb76p-1, 0x1.930b7p-1, 0x1.9cb6aap-1, 0x1.a5fab8p-1, 0x1.aed548p-1,
 0x1.bf4536p-1, 0x1.cdf604p-1, 0x1.dad902p-1, 0x1.e5e15p-1, 0x1.ef03e4p-1,
 0x1.f6379ep-1, 0x1.fb754ap-1, 0x1.feb7aap-1, 0x1.fffb7ep-1, 0x1.ff3f8p-1,
 0x1.fc846ep-1, 0x1.f7cd02p-1, 0x1.f11df2p-1, 0x1.e87deep-1, 0x1.ddf596p-1,
 0x1.d18f6ep-1, 0x1.b35d1ep-1, 0x1.8e5f9cp-1, 0x1.632abp-1, 0x1.326afp-1,
 0x1.f9c63ep-2, 0x1.86d224p-2, 0x1.0dc4cap-2, 0x1.210386p-3, 0x1.0fd77p-6,
 -0x1.bb2ad2p-4, -0x1.d9b092p-3, -0x1.6733b8p-2, -0x1.dbf436p-2,
 -0x1.24a3bp-1, -0x1.56bc3ap-1, -0x1.837b9ep-1, -0x1.ca3c0cp-1,
 -0x1.f47ed4p-1, -0x1.ffa34ep-1, -0x1.eaf82p-1, -0x1.b7c644p-1,
 -0x1.693c94p-1, -0x1.043d28p-1, -0x1.1e1f18p-2, -0x1.0fcddcp-5,
 0x1.b890d4p-3, 0x1.ccd85ap-2, 0x1.50608cp-1, 0x1.a56adcp-1, 0x1.e04188p-1,
 0x1.fd3c0ep-1, 0x1.fa8d2ap-1, 0x1.98d34ep-1, 0x1.a60264p-2, -0x1.33d1aap-4,
 -0x1.1689fp-1, -0x1.c2677cp-1, -0x1.fffeb8p-1, -0x1.c03b44p-1, -0x1.12b9bp-1,
 -0x1.0fa78cp-4, 0x1.ae4044p-2, 0x1.9b89a2p-1, 0x1.fb30e4p-1, 0x1.deaa9p-1,
 0x1.4cf288p-1, 0x1.a6d86cp-3, -0x1.26d02p-2, -0x1.ec3c4ap-1, -0x1.808166p-1,
 0x1.32f2d2p-3, 0x1.d36d9p-1, 0x1.ac5e2p-1, -0x1.220a2ap-7, -0x1.b143cep-1,
 -0x1.cfa7f8p-1, -0x1.0f0e7p-3, 0x1.866e1p-1, 0x1.e9aa1cp-1, 0x1.156854p-2,
 -0x1.53c7d2p-1, -0x1.f9df48p-1, -0x1.9dbc0cp-2, 0x1.1a549ap-1, 0x1.0ee3eep-1,
 -0x1.fbca7p-1, 0x1.2f7b3ep-2, 0x1.7d7f78p-1, -0x1.d5425p-1, 0x1.220742p-6,
 0x1.cdb734p-1, -0x1.8958acp-1, -0x1.0cabfep-2, 0x1.f9274p-1, -0x1.1e199ap-1,
 -0x1.0b08bcp-1, 0x1.fc59cep-1, -0x1.382046p-2, -0x1.7a75e4p-1, 0x1.d70da8p-1,
 -0x1.cbbd2ep-1, 0x1.03ea46p-2, 0x1.21d8dcp-1, -0x1.fcdefap-1, 0x1.7764b8p-1,
 0x1.21fbap-5, -0x1.8f162ep-1, 0x1.f798dp-1, -0x1.03425cp-1, -0x1.495762p-2,
 0x1.da87f4p-1, -0x1.c7ad8p-1, 0x1.e4aebap-3, 0x1.2945dep-1, -0x1.fdcabp-1,
 0x1.712bdap-1, -0x1.94b3aap-1, -0x1.f6ce5ep-2, 0x1.dddc3p-1, 0x1.c16208p-3,
 -0x1.fe8d8p-1, 0x1.21cd18p-4, 0x1.f402dep-1, -0x1.6b74f6p-2, -0x1.bf20d2p-1,
 0x1.37d7cp-1, 0x1.6461c8p-1, -0x1.9f8ccap-1, -0x1.d6e9ecp-2, 0x1.e41164p-1,
 0x1.7a5f52p-3, -0x1.ff9832p-1, 0x1.efcc7cp-1, -0x1.b604acp-1, 0x1.572558p-1,
 -0x1.b66e6p-2, 0x1.32e332p-3, 0x1.21132cp-3, -0x1.ae477p-2, 0x1.53cac6p-1,
 -0x1.b3abb4p-1, 0x1.eea802p-1, -0x1.ffc0fcp-1, 0x1.e583fcp-1, -0x1.a229cep-1,
 0x1.3b668ap-1, -0x1.73dbd4p-2, 0x1.45b53p-4, -0x1.eef1d2p-2, -0x1.c59b9ap-1,
 -0x1.fd5922p-1, -0x1.8c3666p-1, -0x1.2f6c26p-2, 0x1.1e2e46p-2, 0x1.86732cp-1,
 0x1.fc5abep-1, 0x1.c9b664p-1, 0x1.fea0f6p-2, -0x1.fbbe8ep-5, -0x1.3442cap-1,
 -0x1.e29538p-1, -0x1.f0e724p-1, -0x1.5a792p-1, -0x1.44ad26p-3, 0x1.b14a1ep-1,
 0x1.a4be8ep-1, -0x1.9e2f3p-3, -0x1.f5e51ap-1, -0x1.21cbd4p-1, 0x1.12c706p-1,
 0x1.f924acp-1, 0x1.e4714ep-3, -0x1.9a3a22p-1, -0x1.ba9c8ep-1, 0x1.fac4a4p-4,
 0x1.ec41bap-1, 0x1.427156p-1, -0x1.df1b82p-2, -0x1.fe2f38p-1, -0x1.409204p-2,
 0x1.cdacf4p-1, -0x1.df8042p-1, 0x1.95a032p-2, 0x1.8ce3a4p-2, -0x1.ddd0d6p-1,
 0x1.cfb56p-1, -0x1.49933ep-2, -0x1.d6b1c8p-2, 0x1.eaee08p-1, -0x1.bcfaacp-1,
 0x1.f6dfe2p-3, 0x1.0ec266p-1, -0x1.f4ef48p-1, 0x1.a76e88p-1, -0x1.5769fap-3,
};

/* For 0 <= i1 < 2^8, C1[i1] stores the binary32 approximation of cos(x1)
   to nearest, where the bfloat16 encoding of x1 is 123*2^7+i1*2^3,
   Table generated by tableC1() from sin.sage. */
static const float C1[] = {
 0x1.ff0016p-1, 0x1.fedf1cp-1, 0x1.febc22p-1, 0x1.fe972ap-1, 0x1.fe7034p-1,
 0x1.fe474p-1, 0x1.fe1c4cp-1, 0x1.fdef5cp-1, 0x1.fdc06cp-1, 0x1.fd8f8p-1,
 0x1.fd5c94p-1, 0x1.fd27acp-1, 0x1.fcf0c8p-1, 0x1.fcb7e6p-1, 0x1.fc7d08p-1,
 0x1.fc402cp-1, 0x1.fc0156p-1, 0x1.fb7db2p-1, 0x1.faf222p-1, 0x1.fa5ea6p-1,
 0x1.f9c34p-1, 0x1.f91ff4p-1, 0x1.f874c2p-1, 0x1.f7c1bp-1, 0x1.f706bep-1,
 0x1.f643fp-1, 0x1.f57948p-1, 0x1.f4a6ccp-1, 0x1.f3cc7cp-1, 0x1.f2ea5ep-1,
 0x1.f20074p-1, 0x1.f10ecp-1, 0x1.f0154ap-1, 0x1.ee0b2p-1, 0x1.ebe214p-1,
 0x1.e99a4cp-1, 0x1.e733eap-1, 0x1.e4af14p-1, 0x1.e20bf4p-1, 0x1.df4ab4p-1,
 0x1.dc6b7ep-1, 0x1.d96e82p-1, 0x1.d653fp-1, 0x1.d31bf8p-1, 0x1.cfc6dp-1,
 0x1.cc54aap-1, 0x1.c8c5cp-1, 0x1.c51a48p-1, 0x1.c1528p-1, 0x1.b96eeep-1,
 0x1.b11d04p-1, 0x1.a85ed4p-1, 0x1.9f368ep-1, 0x1.95a67ep-1, 0x1.8bb106p-1,
 0x1.8158a4p-1, 0x1.769fecp-1, 0x1.6b899p-1, 0x1.601852p-1, 0x1.544f1p-1,
 0x1.4830bep-1, 0x1.3bc06p-1, 0x1.2f0114p-1, 0x1.21f608p-1, 0x1.14a28p-1,
 0x1.f25ec6p-2, 0x1.b98656p-2, 0x1.7ef484p-2, 0x1.42e3dep-2, 0x1.05906ep-2,
 0x1.8e6f08p-3, 0x1.102ee6p-3, 0x1.21bd54p-4, 0x1.0fd9d6p-7, -0x1.bbd1bp-5,
 -0x1.dcef14p-4, -0x1.6d0c44p-3, -0x1.ea3412p-3, -0x1.32b8eap-2,
 -0x1.6f252ap-2, -0x1.aa2266p-2, -0x1.0d72c8p-1, -0x1.419ffap-1,
 -0x1.70c856p-1, -0x1.9a2f7ep-1, -0x1.bd300cp-1, -0x1.d93e2ap-1,
 -0x1.ede9c6p-1, -0x1.fae04cp-1, -0x1.ffedf6p-1, -0x1.fcfe9p-1,
 -0x1.f21dd8p-1, -0x1.df774p-1, -0x1.c5554ap-1, -0x1.a4205cp-1,
 -0x1.7c5d1ap-1, -0x1.4eaa6p-1, -0x1.c8cb28p-2, -0x1.afb5b6p-3, 0x1.34096ep-5,
 0x1.227858p-2, 0x1.063012p-1, 0x1.6ad6c4p-1, 0x1.b8ee36p-1, 0x1.eb9b7p-1,
 0x1.ffb7d6p-1, 0x1.f4034cp-1, 0x1.c9382p-1, 0x1.81ff7ap-1, 0x1.22c6f6p-1,
 0x1.62f45ep-2, 0x1.a92446p-4, -0x1.29fbecp-3, -0x1.343ae8p-1, -0x1.d27faap-1,
 -0x1.fe8d5ap-1, -0x1.ad9ac8p-1, -0x1.e6f328p-2, 0x1.220ae4p-8, 0x1.eee772p-2,
 0x1.b00dap-1, 0x1.fedf6ap-1, 0x1.d09cdep-1, 0x1.30997p-1, 0x1.1809aep-3,
 -0x1.6b7144p-2, -0x1.84f5dp-1, -0x1.f4f7dap-1, -0x1.ea5258p-1,
 -0x1.19c46cp-2, 0x1.521508p-1, 0x1.fa377ep-1, 0x1.a1e044p-2, -0x1.186ff8p-1,
 -0x1.fffadep-1, -0x1.10cf7ep-1, 0x1.b25bfcp-2, 0x1.fb7efp-1, 0x1.4b3902p-1,
 -0x1.2b267p-2, -0x1.ecdaaep-1, -0x1.7f0166p-1, 0x1.3be83p-3, 0x1.d4591cp-1,
 0x1.ab1f54p-1, -0x1.b277cep-1, -0x1.0611d4p-3, 0x1.e8ff6ap-1, -0x1.5578e8p-1,
 -0x1.9995cp-2, 0x1.ffeb76p-1, -0x1.ba8cdap-2, -0x1.47c1p-1, 0x1.ee1006p-1,
 -0x1.4dce1cp-3, -0x1.a89b4cp-1, 0x1.b4d944p-1, 0x1.e8296ep-4, -0x1.e7a2acp-1,
 0x1.58d5eep-1, 0x1.914306p-2, 0x1.c2b4dap-2, -0x1.ef3b78p-1, 0x1.a60ec2p-1,
 -0x1.c4256ap-4, -0x1.5c2c0ap-1, 0x1.ffaddap-1, -0x1.40bd54p-1,
 -0x1.718584p-3, 0x1.b981dcp-1, -0x1.e4cbe2p-1, 0x1.808596p-2, 0x1.d2e99p-2,
 -0x1.f17488p-1, 0x1.a0dc54p-1, -0x1.7c02e6p-4, -0x1.62c33ep-1,
 -0x1.399fecp-1, 0x1.be0708p-1, 0x1.6fa94ep-2, -0x1.f385bp-1, -0x1.33c1e6p-4,
 0x1.feb786p-1, -0x1.b8988cp-3, -0x1.dea9dep-1, 0x1.f2e154p-2, 0x1.961392p-1,
 -0x1.6f9bbep-1, -0x1.2b1a36p-1, 0x1.c6a5aep-1, 0x1.4d999ep-2, -0x1.f72f9ep-1,
 -0x1.45f746p-5, -0x1.ff1e32p-3, 0x1.091c8p-1, -0x1.7bfe48p-1, 0x1.ceb27p-1,
 -0x1.fa3816p-1, 0x1.fadfbep-1, -0x1.d09b34p-1, 0x1.7efec6p-1, -0x1.0cf3a4p-1,
 0x1.08442ap-2, 0x1.fbfd12p-6, -0x1.451386p-2, 0x1.27700ep-1, -0x1.9351cp-1,
 0x1.dd0c26p-1, -0x1.fe60f2p-1, -0x1.c03868p-1, -0x1.dae7f4p-2, 0x1.a05738p-4,
 0x1.444888p-1, 0x1.e901c2p-1, 0x1.eb993cp-1, 0x1.4b32fcp-1, 0x1.e7eaaap-4,
 -0x1.cae1fp-2, -0x1.bbccd6p-1, -0x1.ff03fep-1, -0x1.98cd5cp-1,
 -0x1.56191cp-2, 0x1.eda9f4p-3, 0x1.78f646p-1, 0x1.f9866ap-1, 0x1.10c576p-1,
 -0x1.23c00ep-1, -0x1.f56b92p-1, -0x1.94e15p-3, 0x1.a617b4p-1, 0x1.b00526p-1,
 -0x1.4e0c7cp-3, -0x1.f17846p-1, -0x1.325cb2p-1, 0x1.015e92p-1, 0x1.fc10f4p-1,
 0x1.199e6ep-2, -0x1.8db63ap-1, -0x1.c4808p-1, 0x1.58a2d4p-4, 0x1.e64394p-1,
 -0x1.bab79cp-2, -0x1.67039ep-2, 0x1.d61e36p-1, -0x1.d7fa82p-1, 0x1.6fe44ep-2,
 0x1.b222bap-2, -0x1.e4c1b6p-1, 0x1.c6b436p-1, -0x1.22bc8ap-2, -0x1.fa81fap-2,
 0x1.f05348p-1, -0x1.b28cb4p-1, 0x1.a77acap-3, 0x1.1fd602p-1, -0x1.f8c028p-1,
};

/* For 0 <= i2 < 2^7, S2[i2] stores the binary32 approximation of sin(x2)
   to nearest, where x2 = 2^(h-11)*l, i2 decomposes into 2^3*h+l,
   0 <= h < 2^4, 0 <= l < 2^3.
   Table generated by tableS2() from sin.sage. */
static const float S2[] = {
 0x0p+0, 0x1.fffffep-12, 0x1.fffffap-11, 0x1.7ffff8p-10, 0x1.ffffeap-10,
 0x1.3fffecp-9, 0x1.7fffdcp-9, 0x1.bfffc6p-9, 0x0p+0, 0x1.fffffap-11,
 0x1.ffffeap-10, 0x1.7fffdcp-9, 0x1.ffffaap-9, 0x1.3fffacp-8, 0x1.7fff7p-8,
 0x1.bfff1cp-8, 0x0p+0, 0x1.ffffeap-10, 0x1.ffffaap-9, 0x1.7fff7p-8,
 0x1.fffeaap-8, 0x1.3ffeb2p-7, 0x1.7ffdcp-7, 0x1.bffc6ep-7, 0x0p+0,
 0x1.ffffaap-9, 0x1.fffeaap-8, 0x1.7ffdcp-7, 0x1.fffaaap-7, 0x1.3ffacap-6,
 0x1.7ff7p-6, 0x1.bff1b6p-6, 0x0p+0, 0x1.fffeaap-8, 0x1.fffaaap-7,
 0x1.7ff7p-6, 0x1.ffeaaap-6, 0x1.3feb2cp-5, 0x1.7fdc02p-5, 0x1.bfc6d8p-5,
 0x0p+0, 0x1.fffaaap-7, 0x1.ffeaaap-6, 0x1.7fdc02p-5, 0x1.ffaaaep-5,
 0x1.3facb2p-4, 0x1.7f701p-4, 0x1.bf1b78p-4, 0x0p+0, 0x1.ffeaaap-6,
 0x1.ffaaaep-5, 0x1.7f701p-4, 0x1.feaaeep-4, 0x1.3eb312p-3, 0x1.7dc102p-3,
 0x1.bc6f84p-3, 0x0p+0, 0x1.ffaaaep-5, 0x1.feaaeep-4, 0x1.7dc102p-3,
 0x1.faaeeep-3, 0x1.3ad12ap-2, 0x1.771026p-2, 0x1.b1d83p-2, 0x0p+0,
 0x1.feaaeep-4, 0x1.faaeeep-3, 0x1.771026p-2, 0x1.eaee88p-2, 0x1.2b91dep-1,
 0x1.5cffc2p-1, 0x1.88fb76p-1, 0x0p+0, 0x1.faaeeep-3, 0x1.eaee88p-2,
 0x1.5cffc2p-1, 0x1.aed548p-1, 0x1.e5e15p-1, 0x1.feb7aap-1, 0x1.f7cd02p-1,
 0x0p+0, 0x1.eaee88p-2, 0x1.aed548p-1, 0x1.feb7aap-1, 0x1.d18f6ep-1,
 0x1.326afp-1, 0x1.210386p-3, -0x1.6733b8p-2, 0x0p+0, 0x1.aed548p-1,
 0x1.d18f6ep-1, 0x1.210386p-3, -0x1.837b9ep-1, -0x1.eaf82p-1, -0x1.1e1f18p-2,
 0x1.50608cp-1, 0x0p+0, 0x1.d18f6ep-1, -0x1.837b9ep-1, -0x1.1e1f18p-2,
 0x1.fa8d2ap-1, -0x1.1689fp-1, -0x1.12b9bp-1, 0x1.fb30e4p-1, 0x0p+0,
 -0x1.837b9ep-1, 0x1.fa8d2ap-1, -0x1.12b9bp-1, -0x1.26d02p-2, 0x1.d36d9p-1,
 -0x1.cfa7f8p-1, 0x1.156854p-2, 0x0p+0, 0x1.fa8d2ap-1, -0x1.26d02p-2,
 -0x1.cfa7f8p-1, 0x1.1a549ap-1, 0x1.7d7f78p-1, -0x1.8958acp-1, -0x1.0b08bcp-1,
 0x0p+0, -0x1.26d02p-2, 0x1.1a549ap-1, -0x1.8958acp-1, 0x1.d70da8p-1,
 -0x1.fcdefap-1, 0x1.f798dp-1, -0x1.c7ad8p-1,
};

/* For 0 <= i2 < 2^7, C2[i2] stores the binary32 approximation of cos(x2)
   to nearest, where x2 = 2^(h-11)*l, i2 decomposes into 2^3*h+l,
   0 <= h < 2^4, 0 <= l < 2^3.
   Table generated by tableC2() from sin.sage. */
static const float C2[] = {
 0x1p+0, 0x1.fffffcp-1, 0x1.fffffp-1, 0x1.ffffdcp-1, 0x1.ffffcp-1,
 0x1.ffff9cp-1, 0x1.ffff7p-1, 0x1.ffff3cp-1, 0x1p+0, 0x1.fffffp-1,
 0x1.ffffcp-1, 0x1.ffff7p-1, 0x1.ffffp-1, 0x1.fffe7p-1, 0x1.fffdcp-1,
 0x1.fffcfp-1, 0x1p+0, 0x1.ffffcp-1, 0x1.ffffp-1, 0x1.fffdcp-1, 0x1.fffcp-1,
 0x1.fff9cp-1, 0x1.fff7p-1, 0x1.fff3cp-1, 0x1p+0, 0x1.ffffp-1, 0x1.fffcp-1,
 0x1.fff7p-1, 0x1.fffp-1, 0x1.ffe7p-1, 0x1.ffdcp-1, 0x1.ffcfp-1, 0x1p+0,
 0x1.fffcp-1, 0x1.fffp-1, 0x1.ffdcp-1, 0x1.ffc002p-1, 0x1.ff9c04p-1,
 0x1.ff7006p-1, 0x1.ff3c0cp-1, 0x1p+0, 0x1.fffp-1, 0x1.ffc002p-1,
 0x1.ff7006p-1, 0x1.ff0016p-1, 0x1.fe7034p-1, 0x1.fdc06cp-1, 0x1.fcf0c8p-1,
 0x1p+0, 0x1.ffc002p-1, 0x1.ff0016p-1, 0x1.fdc06cp-1, 0x1.fc0156p-1,
 0x1.f9c34p-1, 0x1.f706bep-1, 0x1.f3cc7cp-1, 0x1p+0, 0x1.ff0016p-1,
 0x1.fc0156p-1, 0x1.f706bep-1, 0x1.f0154ap-1, 0x1.e733eap-1, 0x1.dc6b7ep-1,
 0x1.cfc6dp-1, 0x1p+0, 0x1.fc0156p-1, 0x1.f0154ap-1, 0x1.dc6b7ep-1,
 0x1.c1528p-1, 0x1.9f368ep-1, 0x1.769fecp-1, 0x1.4830bep-1, 0x1p+0,
 0x1.f0154ap-1, 0x1.c1528p-1, 0x1.769fecp-1, 0x1.14a28p-1, 0x1.42e3dep-2,
 0x1.21bd54p-4, -0x1.6d0c44p-3, 0x1p+0, 0x1.c1528p-1, 0x1.14a28p-1,
 0x1.21bd54p-4, -0x1.aa2266p-2, -0x1.9a2f7ep-1, -0x1.fae04cp-1, -0x1.df774p-1,
 0x1p+0, 0x1.14a28p-1, -0x1.aa2266p-2, -0x1.fae04cp-1, -0x1.4eaa6p-1,
 0x1.227858p-2, 0x1.eb9b7p-1, 0x1.81ff7ap-1, 0x1p+0, -0x1.aa2266p-2,
 -0x1.4eaa6p-1, 0x1.eb9b7p-1, -0x1.29fbecp-3, -0x1.ad9ac8p-1, 0x1.b00dap-1,
 0x1.1809aep-3, 0x1p+0, -0x1.4eaa6p-1, -0x1.29fbecp-3, 0x1.b00dap-1,
 -0x1.ea5258p-1, 0x1.a1e044p-2, 0x1.b25bfcp-2, -0x1.ecdaaep-1, 0x1p+0,
 -0x1.29fbecp-3, -0x1.ea5258p-1, 0x1.b25bfcp-2, 0x1.ab1f54p-1, -0x1.5578e8p-1,
 -0x1.47c1p-1, 0x1.b4d944p-1, 0x1p+0, -0x1.ea5258p-1, 0x1.ab1f54p-1,
 -0x1.47c1p-1, 0x1.914306p-2, -0x1.c4256ap-4, -0x1.718584p-3, 0x1.d2e99p-2,
};

/* For 0 <= i < 123, S3[i] stores the binary32 approximation of
   sin(2^(5+i)) to nearest, with some values slightly changed to
   reduce the number of exceptional cases.
   Table generated by tableS3() from sin.sage */
static const float S3[] = {
 0x1.1a549ap-1, 0x1.d70da8p-1, 0x1.712bdap-1, -0x1.ff9832p-1, 0x1.45b53p-4,
 -0x1.44ad26p-3, -0x1.409204p-2, -0x1.3074e8p-1, -0x1.e98f88p-1,
 -0x1.1eb042p-1, 0x1.db0ffcp-1, 0x1.625668p-1, -0x1.ff8bd8p-1, -0x1.58809cp-4,
 0x1.57481ep-3, 0x1.526cccp-2, 0x1.3f6888p-1, 0x1.f34428p-1, 0x1.ba9f46p-2,
 -0x1.8f22f8p-1, -0x1.f3fa14p-1, 0x1.aedbaep-2, -0x1.86dccap-1,
 -0x1.f8eef8p-1, 0x1.4e67cp-2, -0x1.3c1236p-1, -0x1.f14f92p-1, -0x1.d91302p-2,
 0x1.a39038p-1, 0x1.e0ef36p-1, -0x1.49f2c6p-1, 0x1.f8993cp-1, 0x1.55f456p-2,
 -0x1.425308p-1, -0x1.f4df2p-1, -0x1.9f7122p-2, 0x1.7bb732p-1, 0x1.fd6e84p-1,
 -0x1.97b16cp-3, 0x1.8f8886p-2, 0x1.6fde18p-1, 0x1.ffbaeep-1, -0x1.09c494p-4,
 0x1.093536p-3, 0x1.06f98cp-2, 0x1.fc4f5ep-2, 0x1.b9430ap-1, 0x1.bf996ap-1,
 -0x1.b2a66cp-1, 0x1.cb6f76p-1, -0x1.958b4ep-1, 0x1.ef1a90p-1, -0x1.f8904p-2,
 0x1.b70f5ap-1, 0x1.c3b8b8p-1, -0x1.a94ad8p-1, 0x1.d99a3p-1, -0x1.67e574p-1,
 0x1.fff6ep-1, 0x1.82a354p-6, -0x1.8287c2p-5, -0x1.82198cp-4, -0x1.806172p-3,
 -0x1.798cc4p-2, -0x1.5ef362p-1, -0x1.ff115ep-1, -0x1.ed331ep-4,
 0x1.e99c7ep-3, 0x1.db69f8p-2, 0x1.a5122p-1, 0x1.df1908p-1, -0x1.51f3b8p-1,
 0x1.fbbf0ep-1, 0x1.05417p-2, -0x1.f93824p-2, -0x1.b7728p-1, -0x1.c30288p-1,
 0x1.aaf6c4p-1, -0x1.d74732p-1, 0x1.705fd8p-1, -0x1.ffae48p-1, 0x1.210d0ep-4,
 -0x1.205492p-3, -0x1.1d7566p-2, -0x1.12247ep-1, -0x1.cf119cp-1,
 -0x1.8b1bb8p-1, 0x1.f6923cp-1, -0x1.8005c8p-2, 0x1.63febcp-1, 0x1.ffb824p-1,
 0x1.0f126p-4, -0x1.0e7a3ep-3, -0x1.0c1bb6p-2, -0x1.02c164p-1, -0x1.be8edap-1,
 -0x1.b4e4aep-1, 0x1.c79988p-1, -0x1.9fbfecp-1, 0x1.e54c46p-1, -0x1.355a3p-1,
 0x1.ed0098p-1, 0x1.0a1946p-1, -0x1.c6acaap-1, -0x1.a21844p-1, 0x1.e2a982p-1,
 -0x1.4212dep-1, 0x1.f4bcc8p-1, 0x1.a1cc8p-2, -0x1.7d7108p-1, -0x1.fce3cp-1,
 0x1.c01004p-3, -0x1.b53538p-2, -0x1.8b5aa2p-1, -0x1.f66c46p-1, 0x1.82e346p-2,
 -0x1.663618p-1, -0x1.ffe302p-1, -0x1.5881f6p-5, 0x1.5833f2p-4, 0x1.56fc44p-3,
 0x1.52242cp-2, 0x1.3f2c62p-1,
};

/* For 0 <= i < 123, C3[i] stores the binary32 approximation of
   cos(2^(5+i)) to nearest, with some values slightly changed to
   reduce the number of exceptional cases.
   Table generated by tableC3() from sin.sage */
static const float C3[] = {
 0x1.ab1f54p-1, 0x1.914306p-2, -0x1.62c33ep-1, -0x1.45f746p-5, -0x1.fe60f2p-1,
 0x1.f9866ap-1, 0x1.e64394p-1, 0x1.9ba4a8p-1, 0x1.2bd43ep-2, -0x1.a835a2p-1,
 0x1.7de36ap-2, -0x1.719454p-1, 0x1.58ced6p-5, -0x1.fe2f94p-1, 0x1.f8c198p-1,
 0x1.e33adap-1, 0x1.902722p-1, 0x1.c5e944p-3, -0x1.cdb2cap-1, 0x1.40ad68p-1,
 -0x1.b9381ap-3, -0x1.d078dap-1, 0x1.4ab652p-1, -0x1.5315d4p-3,
 -0x1.e3edd2p-1, 0x1.92cb46p-1, 0x1.e70c2ep-3, -0x1.c61608p-1, 0x1.25723p-1,
 -0x1.5f42d8p-2, -0x1.8781eap-1, 0x1.5af86p-3, -0x1.e29bbap-1, 0x1.8dcecap-1,
 0x1.a8ac1ap-3, -0x1.d3f858p-1, 0x1.5774p-1, -0x1.99bf9ap-4, -0x1.f5c0a8p-1,
 0x1.d76ba6p-1, 0x1.641d4ap-1, -0x1.09e872p-5, -0x1.feebccp-1, 0x1.fbb05ep-1,
 0x1.eed40cp-1, 0x1.bc7712p-1, 0x1.03ad3ap-1, -0x1.f1300ep-2, -0x1.0e9918p-1,
 -0x1.c3f168p-2, -0x1.3888e2p-1, -0x1.04e422p-2, -0x1.bd87f4p-1,
 0x1.0761eep-1, -0x1.e20b48p-2, -0x1.1d146p-1, -0x1.85134p-2, -0x1.6c2b2p-1,
 0x1.82aa38p-7, -0x1.ffdb8p-1, 0x1.ff6e04p-1, 0x1.fdb862p-1, 0x1.f6e6bcp-1,
 0x1.dbedb8p-1, 0x1.74cc06p-1, 0x1.ee1968p-5, -0x1.fc465ap-1, 0x1.f12748p-1,
 0x1.c5798cp-1, 0x1.234766p-1, -0x1.692938p-2, -0x1.809ebp-1, 0x1.0771bep-3,
 -0x1.ef0e52p-1, 0x1.bd586p-1, 0x1.06bc5ep-1, -0x1.e4b3cap-2, -0x1.1a91ecp-1,
 -0x1.903464p-2, -0x1.63970ep-1, -0x1.213b38p-5, -0x1.feb93ap-1,
 0x1.fae686p-1, 0x1.ebb41ap-1, 0x1.b06c5ap-1, 0x1.b4dbb8p-2, -0x1.45a0b6p-1,
 -0x1.873a2ep-3, -0x1.daa1d2p-1, 0x1.6ffbaap-1, 0x1.0f387p-5, -0x1.fee0a8p-1,
 0x1.fb83ep-1, 0x1.ee239ep-1, 0x1.b9cd7ep-1, 0x1.f4eb4p-2, -0x1.0af60ep-1,
 -0x1.d3377ep-2, -0x1.2ad31ep-1, -0x1.465f9cp-2, -0x1.97fa16p-1,
 0x1.145a4ep-2, -0x1.b56b4ap-1, 0x1.d6cefp-2, -0x1.2788ep-1, -0x1.55a682p-2,
 -0x1.8e02cp-1, 0x1.ab3228p-3, -0x1.d371fap-1, 0x1.55890ap-1, -0x1.c2cd1ap-4,
 -0x1.f398a8p-1, 0x1.cefc7ep-1, 0x1.455452p-1, -0x1.8a4336p-3, -0x1.da0cdp-1,
 0x1.6dd376p-1, 0x1.58957ap-6, -0x1.ff8c0cp-1, 0x1.fe3062p-1, 0x1.f8c4d2p-1,
 0x1.e3479p-1, 0x1.90571ep-1,
};
#endif /* ROUND_SIN */

/* ── log(x) tables (from inria-logbf16.c) ───────────────────────────── */
#if defined(ROUND_LOG)
typedef union { float f; uint32_t u; } b32u32_u;   /* for T3's -Inf entry */

/* For 0 <= i1 < 2^8, T1[i1] approximates log(x1), where the 16-bit
   encoding of x1 is i1*2^7 (i1 encodes the exponent of x1).
   Table generated by table1() in log.sage. */
static const float T1[] = {
 0, -0x1.5d58ap+6, -0x1.5a92d6p+6, -0x1.57cd0ep+6, -0x1.550746p+6,
 -0x1.52417ep+6, -0x1.4f7bb6p+6, -0x1.4cb5ecp+6, -0x1.49f024p+6,
 -0x1.472a5cp+6, -0x1.446494p+6, -0x1.419eccp+6, -0x1.3ed904p+6,
 -0x1.3c133ap+6, -0x1.394d72p+6, -0x1.3687aap+6, -0x1.33c1e2p+6,
 -0x1.30fc1ap+6, -0x1.2e365p+6, -0x1.2b7088p+6, -0x1.28aacp+6, -0x1.25e4f8p+6,
 -0x1.231f3p+6, -0x1.205966p+6, -0x1.1d939ep+6, -0x1.1acdd6p+6,
 -0x1.18080ep+6, -0x1.154246p+6, -0x1.127c7ep+6, -0x1.0fb6b4p+6,
 -0x1.0cf0ecp+6, -0x1.0a2b24p+6, -0x1.07655cp+6, -0x1.049f94p+6,
 -0x1.01d9cap+6, -0x1.fe2804p+5, -0x1.f89c74p+5, -0x1.f310e4p+5,
 -0x1.ed8552p+5, -0x1.e7f9c2p+5, -0x1.e26e32p+5, -0x1.dce2ap+5, -0x1.d7571p+5,
 -0x1.d1cb7ep+5, -0x1.cc3feep+5, -0x1.c6b45ep+5, -0x1.c128ccp+5,
 -0x1.bb9d3cp+5, -0x1.b611acp+5, -0x1.b0861ap+5, -0x1.aafa8ap+5,
 -0x1.a56ef8p+5, -0x1.9fe368p+5, -0x1.9a57d8p+5, -0x1.94cc46p+5,
 -0x1.8f40b6p+5, -0x1.89b526p+5, -0x1.842994p+5, -0x1.7e9e04p+5,
 -0x1.791272p+5, -0x1.7386e2p+5, -0x1.6dfb52p+5, -0x1.686fcp+5, -0x1.62e43p+5,
 -0x1.5d58ap+5, -0x1.57cd0ep+5, -0x1.52417ep+5, -0x1.4cb5ecp+5,
 -0x1.472a5cp+5, -0x1.419eccp+5, -0x1.3c133ap+5, -0x1.3687aap+5,
 -0x1.30fc1ap+5, -0x1.2b7088p+5, -0x1.25e4f8p+5, -0x1.205966p+5,
 -0x1.1acdd6p+5, -0x1.154246p+5, -0x1.0fb6b4p+5, -0x1.0a2b24p+5,
 -0x1.049f94p+5, -0x1.fe2804p+4, -0x1.f310e4p+4, -0x1.e7f9c2p+4,
 -0x1.dce2ap+4, -0x1.d1cb7ep+4, -0x1.c6b45ep+4, -0x1.bb9d3cp+4,
 -0x1.b0861ap+4, -0x1.a56ef8p+4, -0x1.9a57d8p+4, -0x1.8f40b6p+4,
 -0x1.842994p+4, -0x1.791272p+4, -0x1.6dfb52p+4, -0x1.62e43p+4,
 -0x1.57cd0ep+4, -0x1.4cb5ecp+4, -0x1.419eccp+4, -0x1.3687aap+4,
 -0x1.2b7088p+4, -0x1.205966p+4, -0x1.154246p+4, -0x1.0a2b24p+4,
 -0x1.fe2804p+3, -0x1.e7f9c2p+3, -0x1.d1cb7ep+3, -0x1.bb9d3cp+3,
 -0x1.a56ef8p+3, -0x1.8f40b6p+3, -0x1.791272p+3, -0x1.62e43p+3,
 -0x1.4cb5ecp+3, -0x1.3687aap+3, -0x1.205966p+3, -0x1.0a2b24p+3,
 -0x1.e7f9c2p+2, -0x1.bb9d3cp+2, -0x1.8f40b6p+2, -0x1.62e43p+2,
 -0x1.3687aap+2, -0x1.0a2b24p+2, -0x1.bb9d3cp+1, -0x1.62e43p+1,
 -0x1.0a2b24p+1, -0x1.62e43p+0, -0x1.62e43p-1, 0x0p+0, 0x1.62e43p-1,
 0x1.62e43p+0, 0x1.0a2b24p+1, 0x1.62e43p+1, 0x1.bb9d3cp+1, 0x1.0a2b24p+2,
 0x1.3687aap+2, 0x1.62e43p+2, 0x1.8f40b6p+2, 0x1.bb9d3cp+2, 0x1.e7f9c2p+2,
 0x1.0a2b24p+3, 0x1.205966p+3, 0x1.3687aap+3, 0x1.4cb5ecp+3, 0x1.62e43p+3,
 0x1.791272p+3, 0x1.8f40b6p+3, 0x1.a56ef8p+3, 0x1.bb9d3cp+3, 0x1.d1cb7ep+3,
 0x1.e7f9c2p+3, 0x1.fe2804p+3, 0x1.0a2b24p+4, 0x1.154246p+4, 0x1.205966p+4,
 0x1.2b7088p+4, 0x1.3687aap+4, 0x1.419eccp+4, 0x1.4cb5ecp+4, 0x1.57cd0ep+4,
 0x1.62e43p+4, 0x1.6dfb52p+4, 0x1.791272p+4, 0x1.842994p+4, 0x1.8f40b6p+4,
 0x1.9a57d8p+4, 0x1.a56ef8p+4, 0x1.b0861ap+4, 0x1.bb9d3cp+4, 0x1.c6b45ep+4,
 0x1.d1cb7ep+4, 0x1.dce2ap+4, 0x1.e7f9c2p+4, 0x1.f310e4p+4, 0x1.fe2804p+4,
 0x1.049f94p+5, 0x1.0a2b24p+5, 0x1.0fb6b4p+5, 0x1.154246p+5, 0x1.1acdd6p+5,
 0x1.205966p+5, 0x1.25e4f8p+5, 0x1.2b7088p+5, 0x1.30fc1ap+5, 0x1.3687aap+5,
 0x1.3c133ap+5, 0x1.419eccp+5, 0x1.472a5cp+5, 0x1.4cb5ecp+5, 0x1.52417ep+5,
 0x1.57cd0ep+5, 0x1.5d58ap+5, 0x1.62e43p+5, 0x1.686fcp+5, 0x1.6dfb52p+5,
 0x1.7386e2p+5, 0x1.791272p+5, 0x1.7e9e04p+5, 0x1.842994p+5, 0x1.89b526p+5,
 0x1.8f40b6p+5, 0x1.94cc46p+5, 0x1.9a57d8p+5, 0x1.9fe368p+5, 0x1.a56ef8p+5,
 0x1.aafa8ap+5, 0x1.b0861ap+5, 0x1.b611acp+5, 0x1.bb9d3cp+5, 0x1.c128ccp+5,
 0x1.c6b45ep+5, 0x1.cc3feep+5, 0x1.d1cb7ep+5, 0x1.d7571p+5, 0x1.dce2ap+5,
 0x1.e26e32p+5, 0x1.e7f9c2p+5, 0x1.ed8552p+5, 0x1.f310e4p+5, 0x1.f89c74p+5,
 0x1.fe2804p+5, 0x1.01d9cap+6, 0x1.049f94p+6, 0x1.07655cp+6, 0x1.0a2b24p+6,
 0x1.0cf0ecp+6, 0x1.0fb6b4p+6, 0x1.127c7ep+6, 0x1.154246p+6, 0x1.18080ep+6,
 0x1.1acdd6p+6, 0x1.1d939ep+6, 0x1.205966p+6, 0x1.231f3p+6, 0x1.25e4f8p+6,
 0x1.28aacp+6, 0x1.2b7088p+6, 0x1.2e365p+6, 0x1.30fc1ap+6, 0x1.33c1e2p+6,
 0x1.3687aap+6, 0x1.394d72p+6, 0x1.3c133ap+6, 0x1.3ed904p+6, 0x1.419eccp+6,
 0x1.446494p+6, 0x1.472a5cp+6, 0x1.49f024p+6, 0x1.4cb5ecp+6, 0x1.4f7bb6p+6,
 0x1.52417ep+6, 0x1.550746p+6, 0x1.57cd0ep+6, 0x1.5a92d6p+6, 0x1.5d58ap+6,
 0x1.601e68p+6, 0,
};

/* For 0 <= i2 < 2^7, T2[i2] approximates log(x2), where 1 <= x2 < 2
   and i2 matches the 7 lower bits of the 16-bit encoding of x2.
   Table generated by table2() in log.sage. */
static const float T2[] = {
 0x0p+0, 0x1.fe02a6p-8, 0x1.fc0a8cp-7, 0x1.7b91bp-6, 0x1.f829bp-6,
 0x1.39e87cp-5, 0x1.77459p-5, 0x1.b42dd8p-5, 0x1.f0a30cp-5, 0x1.16536ep-4,
 0x1.341d7ap-4, 0x1.51b074p-4, 0x1.6f0d28p-4, 0x1.8c345ep-4, 0x1.a926d4p-4,
 0x1.c5e548p-4, 0x1.e27076p-4, 0x1.fec914p-4, 0x1.0d77e8p-3, 0x1.1b72aep-3,
 0x1.29553p-3, 0x1.371fc2p-3, 0x1.44d2b6p-3, 0x1.526e5ep-3, 0x1.5ff308p-3,
 0x1.6d60fep-3, 0x1.7ab89p-3, 0x1.87fa06p-3, 0x1.9525aap-3, 0x1.a23bc2p-3,
 0x1.af3c94p-3, 0x1.bc2868p-3, 0x1.c8ff7cp-3, 0x1.d5c216p-3, 0x1.e27076p-3,
 0x1.ef0adcp-3, 0x1.fb9186p-3, 0x1.04025ap-2, 0x1.0a324ep-2, 0x1.1058cp-2,
 0x1.1675cap-2, 0x1.1c898cp-2, 0x1.22942p-2, 0x1.2895a2p-2, 0x1.2e8e2cp-2,
 0x1.347ddap-2, 0x1.3a64c6p-2, 0x1.404308p-2, 0x1.4618bcp-2, 0x1.4be5fap-2,
 0x1.51aad8p-2, 0x1.576772p-2, 0x1.5d1bdcp-2, 0x1.62c83p-2, 0x1.686c82p-2,
 0x1.6e08eap-2, 0x1.739d8p-2, 0x1.792a56p-2, 0x1.7eaf84p-2, 0x1.842d1ep-2,
 0x1.89a338p-2, 0x1.8f11e8p-2, 0x1.947942p-2, 0x1.99d958p-2, 0x1.9f323ep-2,
 0x1.a4840ap-2, 0x1.a9cecap-2, 0x1.af1294p-2, 0x1.b44f78p-2, 0x1.b9858ap-2,
 0x1.beb4dap-2, 0x1.c3dd7ap-2, 0x1.c8ff7cp-2, 0x1.ce1afp-2, 0x1.d32fe8p-2,
 0x1.d83e72p-2, 0x1.dd46ap-2, 0x1.e24882p-2, 0x1.e74426p-2, 0x1.ec399ep-2,
 0x1.f128f6p-2, 0x1.f6124p-2, 0x1.faf588p-2, 0x1.ffd2ep-2, 0x1.02552ap-1,
 0x1.04bdfap-1, 0x1.0723e6p-1, 0x1.0986f4p-1, 0x1.0be72ep-1, 0x1.0e4498p-1,
 0x1.109f3ap-1, 0x1.12f71ap-1, 0x1.154c3ep-1, 0x1.179eacp-1, 0x1.19ee6cp-1,
 0x1.1c3b82p-1, 0x1.1e85f6p-1, 0x1.20cdcep-1, 0x1.23130ep-1, 0x1.2555bcp-1,
 0x1.2795e2p-1, 0x1.29d38p-1, 0x1.2c0e9ep-1, 0x1.2e4744p-1, 0x1.307d74p-1,
 0x1.32b134p-1, 0x1.34e28ap-1, 0x1.37117cp-1, 0x1.393e0ep-1, 0x1.3b6844p-1,
 0x1.3d9026p-1, 0x1.3fb5b8p-1, 0x1.41d8fep-1, 0x1.43f9fep-1, 0x1.4618bcp-1,
 0x1.48353ep-1, 0x1.4a4f86p-1, 0x1.4c679ap-1, 0x1.4e7d82p-1, 0x1.50913cp-1,
 0x1.52a2d2p-1, 0x1.54b246p-1, 0x1.56bf9ep-1, 0x1.58cadcp-1, 0x1.5ad404p-1,
 0x1.5cdb1ep-1, 0x1.5ee02ap-1, 0x1.60e33p-1,
};

/* For 0 <= i2 < 2^7, T3[i2] approximates log(x2), where x2 is
   subnormal, and i2 equals the 16-bit encoding of x2.
   Table generated by table3() in log.sage. */
static const b32u32_u T3[] = {
 {.u = 0xff800000}, {-0x1.70c11ap+6}, {-0x1.6dfb52p+6}, {-0x1.6c5c2p+6},
 {-0x1.6b358ap+6}, {-0x1.6a510ap+6}, {-0x1.699656p+6}, {-0x1.68f87ep+6},
 {-0x1.686fcp+6}, {-0x1.67f724p+6}, {-0x1.678b4p+6}, {-0x1.6729a8p+6},
 {-0x1.66d08ep+6}, {-0x1.667e98p+6}, {-0x1.6632b4p+6}, {-0x1.65ec0ep+6},
 {-0x1.65a9f8p+6}, {-0x1.656be4p+6}, {-0x1.65315cp+6}, {-0x1.64f9fep+6},
 {-0x1.64c578p+6}, {-0x1.649382p+6}, {-0x1.6463ep+6}, {-0x1.64365ap+6},
 {-0x1.640ac6p+6}, {-0x1.63e0f8p+6}, {-0x1.63b8dp+6}, {-0x1.63922ap+6},
 {-0x1.636cecp+6}, {-0x1.6348fep+6}, {-0x1.632646p+6}, {-0x1.6304b2p+6},
 {-0x1.62e43p+6}, {-0x1.62c4aep+6}, {-0x1.62a61cp+6}, {-0x1.62886cp+6},
 {-0x1.626b94p+6}, {-0x1.624f86p+6}, {-0x1.623436p+6}, {-0x1.62199ep+6},
 {-0x1.61ffbp+6}, {-0x1.61e668p+6}, {-0x1.61cdbap+6}, {-0x1.61b5a2p+6},
 {-0x1.619e18p+6}, {-0x1.618714p+6}, {-0x1.617092p+6}, {-0x1.615a8cp+6},
 {-0x1.6144fep+6}, {-0x1.612fep+6}, {-0x1.611b3p+6}, {-0x1.6106eap+6},
 {-0x1.60f306p+6}, {-0x1.60df86p+6}, {-0x1.60cc62p+6}, {-0x1.60b998p+6},
 {-0x1.60a724p+6}, {-0x1.609504p+6}, {-0x1.608336p+6}, {-0x1.6071b4p+6},
 {-0x1.60607ep+6}, {-0x1.604f9p+6}, {-0x1.603eeap+6}, {-0x1.602e88p+6},
 {-0x1.601e68p+6}, {-0x1.600e88p+6}, {-0x1.5ffee4p+6}, {-0x1.5fef7ep+6},
 {-0x1.5fe054p+6}, {-0x1.5fd16p+6}, {-0x1.5fc2a4p+6}, {-0x1.5fb41ep+6},
 {-0x1.5fa5ccp+6}, {-0x1.5f97acp+6}, {-0x1.5f89bcp+6}, {-0x1.5f7bfep+6},
 {-0x1.5f6e6ep+6}, {-0x1.5f610cp+6}, {-0x1.5f53d4p+6}, {-0x1.5f46cap+6},
 {-0x1.5f39e8p+6}, {-0x1.5f2d3p+6}, {-0x1.5f209ep+6}, {-0x1.5f1436p+6},
 {-0x1.5f07f2p+6}, {-0x1.5efbd4p+6}, {-0x1.5eefdap+6}, {-0x1.5ee402p+6},
 {-0x1.5ed84ep+6}, {-0x1.5eccbcp+6}, {-0x1.5ec14cp+6}, {-0x1.5eb5fcp+6},
 {-0x1.5eaacap+6}, {-0x1.5e9fb8p+6}, {-0x1.5e94c4p+6}, {-0x1.5e89eep+6},
 {-0x1.5e7f36p+6}, {-0x1.5e7498p+6}, {-0x1.5e6a18p+6}, {-0x1.5e5fb2p+6},
 {-0x1.5e5568p+6}, {-0x1.5e4b38p+6}, {-0x1.5e412p+6}, {-0x1.5e3724p+6},
 {-0x1.5e2d3ep+6}, {-0x1.5e2372p+6}, {-0x1.5e19bep+6}, {-0x1.5e102p+6},
 {-0x1.5e069ap+6}, {-0x1.5dfd2ap+6}, {-0x1.5df3dp+6}, {-0x1.5dea8ap+6},
 {-0x1.5de15cp+6}, {-0x1.5dd842p+6}, {-0x1.5dcf3cp+6}, {-0x1.5dc64ap+6},
 {-0x1.5dbd6cp+6}, {-0x1.5db4a2p+6}, {-0x1.5dabecp+6}, {-0x1.5da348p+6},
 {-0x1.5d9ab6p+6}, {-0x1.5d9236p+6}, {-0x1.5d89c8p+6}, {-0x1.5d816cp+6},
 {-0x1.5d7922p+6}, {-0x1.5d70e8p+6}, {-0x1.5d68cp+6}, {-0x1.5d60a8p+6},
};
#endif /* ROUND_LOG */

/* Load x as a 24-bit MPFR value (exact for any binary32), then round
 * to 16 bits with round-to-nearest-even.  Returns the result as float. */
static float round_to_16bitp(float x)
{
    mpfr_t mp24, mp16;
    mpfr_init2(mp24, 24);
    mpfr_init2(mp16, 16);
    mpfr_set_flt(mp24, x, MPFR_RNDN);  /* exact: float32 has 24-bit sig */
    mpfr_set(mp16, mp24, MPFR_RNDN);   /* round 24 → 16 bits */
    float result = mpfr_get_flt(mp16, MPFR_RNDN);
    mpfr_clear(mp16);
    mpfr_clear(mp24);

    /* The MAX_FLT cap (0x1.fffffep+127) rounds up to 2^128 at 16-bit
     * precision, which overflows binary32 to +Inf.  Mirror the table's
     * "capped to MAX_FLT" intent by clamping to the largest finite value
     * representable in 16-bit precision (0x1.fffep+127). */
    if (isinf(result))
        result = 0x1.fffep+127f;
    return result;
}

/* C-array hex literal in the same style as the source tables
 * (lowercase %a, trailing zeros trimmed: e.g. 0x1.008p+0, 0x1p+0). */
static void format_hex(char *buf, size_t bufsz, float v)
{
    snprintf(buf, bufsz, "%a", (double)v);
}

/* Emit the rounded table as a paste-ready C array, matching the source
 * format: leading space, "value," tokens joined by spaces, greedy-wrapped
 * so no line exceeds 78 columns (the Inria generator's width). */
static void print_c_table(FILE *out, const char *name, const char *comment,
                          const float *tbl, size_t n)
{
    fprintf(out, "%s", comment);
    fprintf(out, "static const float %s[] = {\n", name);

    char line[128];
    size_t line_len = 0;
    for (size_t i = 0; i < n; i++) {
        char tok[64], hex[48];
        format_hex(hex, sizeof hex, round_to_16bitp(tbl[i]));
        snprintf(tok, sizeof tok, "%s,", hex);

        size_t added = 1 + strlen(tok);   /* " " + "value," */
        if (line_len != 0 && line_len + added > 78) {
            fprintf(out, "%s\n", line);
            line_len = 0;
        }
        line_len += snprintf(line + line_len, sizeof(line) - line_len,
                             " %s", tok);
    }
    if (line_len != 0)
        fprintf(out, "%s\n", line);
    fprintf(out, "};\n\n");
}

static void print_table(FILE *out, const char *name,
                        const float *tbl, size_t n)
{
    fprintf(out, "=== %s (%zu entries) ===\n", name, n);
    fprintf(out, "%-9s  %-24s  %-24s  changed\n",
            "index", "original (24-bit)", "rounded (16-bit)");
    fprintf(out, "%-9s  %-24s  %-24s  -------\n",
            "---------", "------------------------", "------------------------");

    int changed = 0;
    for (size_t i = 0; i < n; i++) {
        float orig    = tbl[i];
        float rounded = round_to_16bitp(orig);
        int differs   = (orig != rounded);
        if (differs) ++changed;
        fprintf(out, "%s[%-4zu]  %+-24a  %+-24a  %s\n",
                name, i,
                (double)orig, (double)rounded,
                differs ? "YES" : "");
    }
    fprintf(out, "--- %d of %zu entries changed\n\n", changed, n);
}

#if defined(ROUND_LOG)
/* T3 is a union table whose first entry is -Inf (log of +0).  Round the
 * finite entries; keep the -Inf entry verbatim. */
static void print_table_b32(FILE *out, const char *name,
                            const b32u32_u *tbl, size_t n)
{
    fprintf(out, "=== %s (%zu entries) ===\n", name, n);
    fprintf(out, "%-9s  %-24s  %-24s  changed\n",
            "index", "original (24-bit)", "rounded (16-bit)");
    fprintf(out, "%-9s  %-24s  %-24s  -------\n",
            "---------", "------------------------", "------------------------");

    int changed = 0;
    for (size_t i = 0; i < n; i++) {
        float orig = tbl[i].f;
        if (isinf(orig)) {
            fprintf(out, "%s[%-4zu]  %-24s  %-24s\n", name, i, "-inf", "-inf");
            continue;
        }
        float rounded = round_to_16bitp(orig);
        int differs   = (orig != rounded);
        if (differs) ++changed;
        fprintf(out, "%s[%-4zu]  %+-24a  %+-24a  %s\n",
                name, i, (double)orig, (double)rounded, differs ? "YES" : "");
    }
    fprintf(out, "--- %d of %zu entries changed\n\n", changed, n);
}

static void print_c_table_b32(FILE *out, const char *name, const char *comment,
                              const b32u32_u *tbl, size_t n)
{
    fprintf(out, "%s", comment);
    fprintf(out, "static const b32u32_u %s[] = {\n", name);

    char line[128];
    size_t line_len = 0;
    for (size_t i = 0; i < n; i++) {
        char tok[80], hex[48];
        if (isinf(tbl[i].f))
            snprintf(tok, sizeof tok, "{.u = 0x%08x},", tbl[i].u);
        else {
            format_hex(hex, sizeof hex, round_to_16bitp(tbl[i].f));
            snprintf(tok, sizeof tok, "{%s},", hex);
        }
        size_t added = 1 + strlen(tok);
        if (line_len != 0 && line_len + added > 78) {
            fprintf(out, "%s\n", line);
            line_len = 0;
        }
        line_len += snprintf(line + line_len, sizeof(line) - line_len,
                             " %s", tok);
    }
    if (line_len != 0)
        fprintf(out, "%s\n", line);
    fprintf(out, "};\n\n");
}
#endif /* ROUND_LOG */

int main(void)
{
#if defined(ROUND_SIN)
    FILE *out = fopen("sin/round-16bitp-sin.txt", "w");
    if (!out) { perror("fopen sin/round-16bitp-sin.txt"); return 1; }

    const size_t n_s1 = sizeof(S1) / sizeof(float);
    const size_t n_c1 = sizeof(C1) / sizeof(float);
    const size_t n_s2 = sizeof(S2) / sizeof(float);
    const size_t n_c2 = sizeof(C2) / sizeof(float);
    const size_t n_s3 = sizeof(S3) / sizeof(float);
    const size_t n_c3 = sizeof(C3) / sizeof(float);

    fprintf(out,
        "S/C sin(x) tables from inria-sinbf16.c rounded to 16-bit MPFR precision\n"
        "MPFR version : %s\n"
        "Rounding mode: MPFR_RNDN (nearest, ties-to-even)\n"
        "Method       : set 24-bit MPFR from float (exact), round to 16-bit MPFR\n\n",
        mpfr_get_version());

    print_table(out, "S1", S1, n_s1);
    print_table(out, "C1", C1, n_c1);
    print_table(out, "S2", S2, n_s2);
    print_table(out, "C2", C2, n_c2);
    print_table(out, "S3", S3, n_s3);
    print_table(out, "C3", C3, n_c3);

    fprintf(out,
        "\n================================================================\n"
        " PASTE-READY TABLES (16-bit-rounded values, source format)\n"
        "================================================================\n\n");

    print_c_table(out, "S1",
        "/* sin(x1) table S1, rounded to 16-bit precision. */\n", S1, n_s1);
    print_c_table(out, "C1",
        "/* cos(x1) table C1, rounded to 16-bit precision. */\n", C1, n_c1);
    print_c_table(out, "S2",
        "/* sin(x2) table S2, rounded to 16-bit precision. */\n", S2, n_s2);
    print_c_table(out, "C2",
        "/* cos(x2) table C2, rounded to 16-bit precision. */\n", C2, n_c2);
    print_c_table(out, "S3",
        "/* sin(2^(5+i)) table S3, rounded to 16-bit precision. */\n", S3, n_s3);
    print_c_table(out, "C3",
        "/* cos(2^(5+i)) table C3, rounded to 16-bit precision. */\n", C3, n_c3);

    fclose(out);
    printf("Done.  sin S/C tables rounded.  See sin/round-16bitp-sin.txt\n");
    return 0;

#elif defined(ROUND_LOG)
    FILE *out = fopen("log/round-16bitp-log.txt", "w");
    if (!out) { perror("fopen log/round-16bitp-log.txt"); return 1; }

    const size_t n_t1 = sizeof(T1) / sizeof(float);
    const size_t n_t2 = sizeof(T2) / sizeof(float);
    const size_t n_t3 = sizeof(T3) / sizeof(b32u32_u);

    fprintf(out,
        "T1/T2/T3 log(x) tables from inria-logbf16.c rounded to 16-bit MPFR precision\n"
        "MPFR version : %s\n"
        "Rounding mode: MPFR_RNDN (nearest, ties-to-even)\n"
        "Method       : set 24-bit MPFR from float (exact), round to 16-bit MPFR\n\n",
        mpfr_get_version());

    print_table(out, "T1", T1, n_t1);
    print_table(out, "T2", T2, n_t2);
    print_table_b32(out, "T3", T3, n_t3);

    fprintf(out,
        "\n================================================================\n"
        " PASTE-READY TABLES (16-bit-rounded values, source format)\n"
        "================================================================\n\n");

    print_c_table(out, "T1",
        "/* log(x1) table T1, rounded to 16-bit precision. */\n", T1, n_t1);
    print_c_table(out, "T2",
        "/* log(x2) table T2, rounded to 16-bit precision. */\n", T2, n_t2);
    print_c_table_b32(out, "T3",
        "/* log(subnormal x2) table T3 (T3[0]=-Inf), rounded to 16-bit precision. */\n",
        T3, n_t3);

    fclose(out);
    printf("Done.  log T1/T2/T3 tables rounded.  See log/round-16bitp-log.txt\n");
    return 0;

#else /* ROUND_EXP */
    FILE *out = fopen("round-16bitp.txt", "w");
    if (!out) { perror("fopen round-16bitp.txt"); return 1; }

    const size_t n_t1 = sizeof(T1) / sizeof(float);
    const size_t n_t2 = sizeof(T2) / sizeof(float);

    fprintf(out,
        "T1/T2 values from inria-expbf16.c rounded to 16-bit MPFR precision\n"
        "MPFR version : %s\n"
        "Rounding mode: MPFR_RNDN (nearest, ties-to-even)\n"
        "Method       : set 24-bit MPFR from float (exact), round to 16-bit MPFR\n\n",
        mpfr_get_version());

    print_table(out, "T1", T1, n_t1);
    print_table(out, "T2", T2, n_t2);

    /* Paste-ready C arrays of the 16-bit-rounded values, in the exact
     * source format so they can replace the originals directly. */
    fprintf(out,
        "\n================================================================\n"
        " PASTE-READY TABLES (16-bit-rounded values, source format)\n"
        "================================================================\n\n");

    static const char t1_comment[] =
        "/* For 0 <= i1 < 2^8, T1[i1] stores the binary32 approximation of y1=exp(x1)\n"
        "   to nearest, where the bfloat16 encoding of x1 is (118*2^4+i1)*2^3,\n"
        "   and T1[2^8+i1] stores the binary32 approximation of y1=exp(-x1)\n"
        "   to nearest. Values larger than MAX_FLT are capped to MAX_FLT.\n"
        "   For example for i1=5, we have x1=0x1.5p-9, which yields 0x1.00a838p+0.\n"
        "   Table generated by table1() from exp.sage. */\n";
    static const char t2_comment[] =
        "/* For 0 <= i2 < 2^7, T2[i2] stores the binary32 approximation of y2=exp(x2)\n"
        "   to nearest, where x2 = 2^(h-16)*l, i2 decomposes into 2^3*h+l,\n"
        "   0 <= h < 2^4, 0 <= l < 2^3, and T2[2^7+i2] stores the binary32\n"
        "   approximation of y2=exp(-x2) to nearest.\n"
        "   One entry was manually optimized to avoid an exception.\n"
        "   Table generated by table2() from exp.sage. */\n";

    print_c_table(out, "T1", t1_comment, T1, n_t1);
    print_c_table(out, "T2", t2_comment, T2, n_t2);

    fclose(out);
    printf("Done.  T1: %zu entries, T2: %zu entries.  See round-16bitp.txt\n",
           n_t1, n_t2);
    return 0;
#endif
}