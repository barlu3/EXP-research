/* Diagnostics for the ln 2x2 limb search -- the numbers docs/limb-tuning.tex
   quotes that limb-gen.c does not itself print.

   Reports: exact-midpoint landings; the per-column margin mechanism behind the
   row-126 stall; whether the sweep order matters; how many seeds reach zero;
   and whether seeding T2 columns or displacing limb 0 would help.

   Shares the split, scoring and descent code with table-gen/log/limb-gen.c by
   construction, not by inclusion -- it is a replica, so a divergence between
   the two is itself a finding. Run from the repo root.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpfr.h>
#define WP 300
#define N1 256
#define N2 128
#define TW 16
typedef union { __bf16 f; uint16_t u; } b16;
static double rb(mpfr_t v){b16 t;t.f=(__bf16)mpfr_get_d(v,MPFR_RNDN);return (double)t.f;}
static double st(double v,int n){b16 t;t.f=(__bf16)v;
  int32_t k=(t.u&0x8000)?-(int32_t)(t.u&0x7fff):(int32_t)(t.u&0x7fff);k+=n;
  b16 r;r.u=(k<0)?(uint16_t)(0x8000|(uint32_t)(-k)):(uint16_t)k;return (double)r.f;}
static void sp(mpfr_t id,double*l,int n,mpfr_t r){mpfr_set(r,id,MPFR_RNDN);
  for(int k=0;k<n;k++){l[k]=rb(r);mpfr_sub_d(r,r,l[k],MPFR_RNDN);} }
static double t1[N1][2],t2[N2][2],c1[N1][2],c2[N2][2];
static uint16_t ref[N1][N2];
static uint16_t ev(int i,int j){float s=0;s+=(float)(__bf16)t1[i][1];s+=(float)(__bf16)t1[i][0];
  s+=(float)(__bf16)t2[j][1];s+=(float)(__bf16)t2[j][0];b16 o;o.f=(__bf16)s;return o.u;}
static float evf(int i,int j){float s=0;s+=(float)(__bf16)t1[i][1];s+=(float)(__bf16)t1[i][0];
  s+=(float)(__bf16)t2[j][1];s+=(float)(__bf16)t2[j][0];return s;}
static int rbad(int i){int b=0;for(int j=0;j<N2;j++)if(ev(i,j)!=ref[i][j])b++;return b;}
static int cbad(int j){int b=0;for(int i=1;i<N1-1;i++)if(ev(i,j)!=ref[i][j])b++;return b;}
static int tbad(void){int b=0;for(int i=1;i<N1-1;i++)b+=rbad(i);return b;}
/* one sweep; `order`=0 rows-then-cols (shipped), 1 cols-then-rows (transposed) */
static int sweep(int pin,int order){int ch=0;
  for(int ph=0;ph<2;ph++){ int rows = (order==0)? (ph==0) : (ph==1);
    if(rows){ for(int i=1;i<N1-1;i++){ if(i==pin)continue; int base=rbad(i); if(!base)continue;
        double k=t1[i][1];int bd=0,best=base;
        for(int d=-TW;d<=TW;d++){if(!d)continue;t1[i][1]=st(k,d);int b=rbad(i);
          if(b<best||(b==best&&abs(d)<abs(bd))){best=b;bd=d;}}
        t1[i][1]=bd?st(k,bd):k; if(bd)ch=1; } }
    else { for(int j=0;j<N2;j++){ int base=cbad(j); if(!base)continue;
        double k=t2[j][1];int bd=0,best=base;
        for(int d=-TW;d<=TW;d++){if(!d)continue;t2[j][1]=st(k,d);int b=cbad(j);
          if(b<best||(b==best&&abs(d)<abs(bd))){best=b;bd=d;}}
        t2[j][1]=bd?st(k,bd):k; if(bd)ch=1; } } }
  return ch;}
static void desc(int pin,int order){for(int p=0;p<12&&sweep(pin,order);p++){}}
static void reset(void){memcpy(t1,c1,sizeof t1);memcpy(t2,c2,sizeof t2);}
static double ulpat(double v){if(v==0)return ldexp(1.0,-133);int e;frexp(fabs(v),&e);
  double u=ldexp(1.0,(e-1)-7);return u<ldexp(1.0,-133)?ldexp(1.0,-133):u;}
int main(void){
  mpfr_t l2,id,r,x,lg;mpfr_init2(l2,WP);mpfr_init2(id,WP);mpfr_init2(r,WP);
  mpfr_init2(x,WP);mpfr_init2(lg,WP);mpfr_const_log2(l2,MPFR_RNDN);
  for(int k=1;k<N1-1;k++){mpfr_mul_si(id,l2,k-127,MPFR_RNDN);sp(id,c1[k],2,r);}
  for(int k=0;k<N2;k++){mpfr_set_si(id,128+k,MPFR_RNDN);mpfr_div_si(id,id,128,MPFR_RNDN);
    mpfr_log(id,id,MPFR_RNDN);sp(id,c2[k],2,r);}
  for(int i=1;i<N1-1;i++)for(int j=0;j<N2;j++){b16 xb;xb.u=(uint16_t)(i*128+j);
    mpfr_set_d(x,(double)(float)xb.f,MPFR_RNDN);mpfr_log(lg,x,MPFR_RNDN);
    mpfr_t rr;mpfr_init2(rr,8);mpfr_set(rr,lg,MPFR_RNDN);
    b16 o;o.f=(__bf16)mpfr_get_d(rr,MPFR_RNDN);ref[i][j]=o.u;mpfr_clear(rr);}
  reset();

  /* Q7: exact-midpoint landings in the CANONICAL table */
  int ties=0,tiebad=0;
  for(int i=1;i<N1-1;i++)for(int j=0;j<N2;j++){
    double y=(double)evf(i,j); double u=ulpat(y); double m=y/u, f=m-floor(m);
    if(fabs(f-0.5)==0.0){ ties++; if(ev(i,j)!=ref[i][j]) tiebad++; } }
  printf("=== Q7: exact bf16-midpoint landings, CANONICAL 2x2 ===\n");
  printf("  inputs landing exactly on a midpoint: %d ; of those, misrounded: %d\n",ties,tiebad);
  printf("  (so ties are common; only the ones where even-rounding goes the wrong way fail)\n");

  /* Q8: mechanism at row 126 -- per-column margin, and what a -1 ULP shift does */
  printf("\n=== Q8: why d=-1 on row 126 breaks exactly column 77 ===\n");
  { double shift = st(c1[126][1],-1)-c1[126][1];
    printf("  shift applied to the row sum: %+.6g  (= -2^-17)\n", shift);
    printf("  columns whose signed margin to the boundary is smaller than |shift|:\n");
    for(int j=0;j<N2;j++){
      double y=(double)evf(126,j); double u=ulpat(y);
      double m=y/u, f=m-floor(m); double dm=f-0.5;      /* signed dist to midpoint */
      double marg=dm*u;                                  /* absolute, signed */
      if(fabs(marg)<=fabs(shift)*1.5)
        printf("    col %3d: yhat=%+.9g  signed margin to midpoint=%+.4g (%.3f ulp)%s\n",
               j,y,marg,dm, (j==77||j==126)?"   <<":""); }
  }

  /* Q9: transposed sweep order */
  printf("\n=== Q9: does the sweep order matter? ===\n");
  reset(); desc(-1,0); int a=tbad();
  reset(); desc(-1,1); int b=tbad();
  printf("  rows-then-cols (shipped): %d unsolved after pass 1\n",a);
  printf("  cols-then-rows (transposed): %d unsolved after pass 1\n",b);

  /* Q10: how many (seed row, seed d) pairs reach zero? */
  printf("\n=== Q10: uniqueness of the seeded solution ===\n");
  { reset(); desc(-1,0);
    static double s1[N1][2],s2[N2][2]; memcpy(s1,t1,sizeof t1);memcpy(s2,t2,sizeof t2);
    int found=0;
    for(int i=1;i<N1-1;i++){
      for(int m=1;m<=TW;m++) for(int sg=-1;sg<=1;sg+=2){ int d=sg*m;
        memcpy(t1,s1,sizeof t1);memcpy(t2,s2,sizeof t2);
        t1[i][1]=st(t1[i][1],d); desc(i,0);
        if(!tbad()){ int nt=0;
          for(int k=1;k<N1-1;k++) if(c1[k][1]!=t1[k][1]) nt++;
          for(int k=0;k<N2;k++)   if(c2[k][1]!=t2[k][1]) nt++;
          if(found<12) printf("    seed T1[%d] %+d ULP -> solves, %d entries tuned\n",i,d,nt);
          found++; } }
    }
    printf("  total (row,d) seeds in +-%d that reach zero: %d\n",TW,found);
  }

  /* Q17a: seeding T2 columns as well */
  printf("\n=== Q17a: seeding T2 columns (currently never done) ===\n");
  { reset(); desc(-1,0);
    static double s1[N1][2],s2[N2][2]; memcpy(s1,t1,sizeof t1);memcpy(s2,t2,sizeof t2);
    int found=0,firsti=-1,firstd=0;
    for(int j=0;j<N2;j++) for(int m=1;m<=TW;m++) for(int sg=-1;sg<=1;sg+=2){ int d=sg*m;
      memcpy(t1,s1,sizeof t1);memcpy(t2,s2,sizeof t2);
      t2[j][1]=st(t2[j][1],d);
      /* pinned descent over rows only */
      for(int p=0;p<12;p++){ int ch=0;
        for(int i=1;i<N1-1;i++){ int base=rbad(i); if(!base)continue;
          double k=t1[i][1];int bd=0,best=base;
          for(int dd=-TW;dd<=TW;dd++){if(!dd)continue;t1[i][1]=st(k,dd);int bb=rbad(i);
            if(bb<best||(bb==best&&abs(dd)<abs(bd))){best=bb;bd=dd;}}
          t1[i][1]=bd?st(k,bd):k; if(bd)ch=1; }
        if(!ch)break; }
      if(!tbad()){ if(found==0){firsti=j;firstd=d;} found++; } }
    printf("  column seeds that reach zero: %d", found);
    if(found) printf("  (first: T2[%d] %+d ULP)",firsti,firstd);
    printf("\n");
  }

  /* Q17b: also displacing limb 0 */
  printf("\n=== Q17b: does displacing limb 0 ever help? ===\n");
  { reset(); desc(-1,0);
    int i=-1; for(int k=1;k<N1-1;k++) if(rbad(k)){i=k;break;}
    printf("  stalled row = T1[%d]; scanning (d0,d1) over +-%d x +-%d = %d probes\n",
           i,TW,TW,(2*TW+1)*(2*TW+1));
    double k0=t1[i][0],k1=t1[i][1]; int best=rbad(i),nb=0,b0=0,b1=0;
    for(int d0=-TW;d0<=TW;d0++) for(int d1=-TW;d1<=TW;d1++){
      t1[i][0]=d0?st(k0,d0):k0; t1[i][1]=d1?st(k1,d1):k1;
      int b=rbad(i); if(b<best){best=b;b0=d0;b1=d1;nb=1;} }
    t1[i][0]=k0;t1[i][1]=k1;
    printf("  best row score over the 2-D window: %d (base was %d)%s\n",
           best,rbad(i), nb?"":"  -- no (d0,d1) beats the base");
    printf("  ulp(limb0)=%.4g  ulp(limb1)=%.4g  ratio=%.0fx ; rounding interval=%.4g\n",
           ulpat(c1[i][0]),ulpat(c1[i][1]),ulpat(c1[i][0])/ulpat(c1[i][1]),
           ulpat((double)evf(i,126)));
  }
  return 0;
}
