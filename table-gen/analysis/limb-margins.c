/* Margin and cancellation measurements for the shipped ln limb tables.
   Answers: how much of a bf16 ULP the float32 accumulation error can consume,
   whether the rounding regime and the cancellation regime overlap, and how
   close the tightest-passing input actually is to a rounding boundary.
   Run from the repo root.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <mpfr.h>
#include "implementations/log/logbf16-limb.h"
#define WP 300
typedef union { __bf16 f; uint16_t u; } b16;
typedef union { float f; uint32_t u; } f32;

/* ulp of the bf16 grid at magnitude |v| (v a real, normal range) */
static double bf16_ulp(double v){
  v = fabs(v); if (v==0) return ldexp(1.0,-133);
  int e; frexp(v,&e);            /* v in [2^(e-1), 2^e) -> binade exponent e-1 */
  double u = ldexp(1.0, (e-1)-7);
  return u < ldexp(1.0,-133) ? ldexp(1.0,-133) : u;
}
int main(void){
  mpfr_t x,lg; mpfr_init2(x,WP); mpfr_init2(lg,WP);
  double worst_rel_delta=0, worst_abs_over_ulp=0; int wi1=0,wi2=0;
  int rounds=0, tot=0;
  /* margin = distance from yhat to nearest bf16 midpoint, in units of ulp_bf16 */
  double best[6]; int bi1[6],bi2[6];
  for(int k=0;k<6;k++){best[k]=1e300;bi1[k]=bi2[k]=-1;}
  int exact_tie_tuned=0;
  double cancel_max_when_rounding=0;   /* max cancellation factor among inputs that round */
  double cancel_max_overall=0;

  for(int i1=1;i1<255;i1++) for(int i2=0;i2<128;i2++){
    tot++;
    /* float32 accumulation, exactly as the kernel does it */
    float s=0.0f;
    for(int k=LOGBF16_T1_LIMBS-1;k>=0;k--) s+=(float)T1L[i1][k];
    for(int k=LOGBF16_T2_LIMBS-1;k>=0;k--) s+=(float)T2L[i2][k];
    /* exact sum of the same four limbs, in double */
    double e = (double)(float)T1L[i1][0]+(double)(float)T1L[i1][1]
             + (double)(float)T2L[i2][0]+(double)(float)T2L[i2][1];
    double delta = (double)s - e;
    double u = bf16_ulp(e);
    /* cancellation factor: |largest operand| / |result| */
    double big = fabs((double)(float)T1L[i1][0]);
    double c = fabs(e)>0 ? big/fabs(e) : 0;
    if (c>cancel_max_overall) cancel_max_overall=c;
    if (delta!=0.0){ rounds++;
      if (c>cancel_max_when_rounding) cancel_max_when_rounding=c;
      double r = fabs(delta)/fabs(e);
      if (r>worst_rel_delta) worst_rel_delta=r;
      double ao = fabs(delta)/u;
      if (ao>worst_abs_over_ulp){ worst_abs_over_ulp=ao; wi1=i1; wi2=i2; }
    }
    /* margin to nearest midpoint, in ulp units, using the float32 result */
    double y=(double)s;
    double m = y/u;                       /* grid coordinate */
    double frac = m - floor(m);           /* 0..1 ; midpoints at .5 */
    double d = fabs(frac-0.5);            /* 0 => exactly on a midpoint */
    if (d==0.0) exact_tie_tuned++;
    for(int k=0;k<6;k++) if(d<best[k]){
      for(int q=5;q>k;q--){best[q]=best[q-1];bi1[q]=bi1[q-1];bi2[q]=bi2[q-1];}
      best[k]=d;bi1[k]=i1;bi2[k]=i2;break; }
  }
  printf("=== Q5: accumulation error vs cancellation (SHIPPED tuned tables) ===\n");
  printf("  inputs where the float32 accumulation rounds : %d of %d (%.1f%%)\n",rounds,tot,100.0*rounds/tot);
  printf("  max |delta|/|sum|                            : %.4g\n", worst_rel_delta);
  printf("  max |delta| / ulp_bf16(sum)                  : %.6g   at (%d,%d)\n",
         worst_abs_over_ulp, wi1, wi2);
  printf("  max cancellation factor, all inputs          : %.1fx\n", cancel_max_overall);
  printf("  max cancellation factor, inputs that ROUND   : %.1fx\n", cancel_max_when_rounding);
  printf("\n=== Q6/Q7: margin to nearest bf16 midpoint (tuned tables) ===\n");
  printf("  inputs landing EXACTLY on a midpoint         : %d\n", exact_tie_tuned);
  printf("  five smallest margins (in ulp_bf16 units, 0.5 = maximally safe):\n");
  for(int k=0;k<6 && bi1[k]>=0;k++){
    b16 xb; xb.u=(uint16_t)(bi1[k]*128+bi2[k]);
    printf("    (%3d,%3d) x=%-14.8g margin=%.6g ulp\n",bi1[k],bi2[k],(double)(float)xb.f,best[k]);
  }
  return 0;
}
