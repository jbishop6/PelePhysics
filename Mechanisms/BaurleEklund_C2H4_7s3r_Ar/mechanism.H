#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  C
1  H
2  O
3  N
4  He
5  Ar
*/

// Species
#define C2H4_ID 0
#define H2_ID 1
#define O2_ID 2
#define H2O_ID 3
#define CO_ID 4
#define CO2_ID 5
#define N2_ID 6
#define HE_ID 7
#define AR_ID 8

#define NUM_ELEMENTS 6
#define NUM_SPECIES 9
#define NUM_IONS 0
#define NUM_REACTIONS 3

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp
void atomicWeight(amrex::Real* awt);
//  MISC
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);
void CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real* x,
  amrex::Real* q_f,
  amrex::Real* q_r);
void progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T);
//  SPARSE INFORMATION
void SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP);
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS);
void SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
  mm = 6;
  kk = 9;
  ii = 3;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[9] = {
  0.0356455407428531, // C2H4
  0.4960317460317460, // H2
  0.0312519532470779, // O2
  0.0555092978073827, // H2O
  0.0357015351660121, // CO
  0.0227226249176305, // CO2
  0.0356964374955379, // N2
  0.2498374807187924, // HE
  0.0250312891113892, // AR
};
#endif
const amrex::Real h_global_imw[9] = {
  0.0356455407428531, // C2H4
  0.4960317460317460, // H2
  0.0312519532470779, // O2
  0.0555092978073827, // H2O
  0.0357015351660121, // CO
  0.0227226249176305, // CO2
  0.0356964374955379, // N2
  0.2498374807187924, // HE
  0.0250312891113892, // AR
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[9] = {
  28.054000, // C2H4
  2.016000,  // H2
  31.998000, // O2
  18.015000, // H2O
  28.010000, // CO
  44.009000, // CO2
  28.014000, // N2
  4.002602,  // HE
  39.950000, // AR
};
#endif
const amrex::Real h_global_mw[9] = {
  28.054000, // C2H4
  2.016000,  // H2
  31.998000, // O2
  18.015000, // H2O
  28.010000, // CO
  44.009000, // CO2
  28.014000, // N2
  4.002602,  // HE
  39.950000, // AR
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 0.0356455407428531; // C2H4
  imw_new[1] = 0.4960317460317460; // H2
  imw_new[2] = 0.0312519532470779; // O2
  imw_new[3] = 0.0555092978073827; // H2O
  imw_new[4] = 0.0357015351660121; // CO
  imw_new[5] = 0.0227226249176305; // CO2
  imw_new[6] = 0.0356964374955379; // N2
  imw_new[7] = 0.2498374807187924; // HE
  imw_new[8] = 0.0250312891113892; // AR
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
imw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_mw(amrex::Real* mw_new)
{
  mw_new[0] = 28.054000; // C2H4
  mw_new[1] = 2.016000;  // H2
  mw_new[2] = 31.998000; // O2
  mw_new[3] = 18.015000; // H2O
  mw_new[4] = 28.010000; // CO
  mw_new[5] = 44.009000; // CO2
  mw_new[6] = 28.014000; // N2
  mw_new[7] = 4.002602;  // HE
  mw_new[8] = 39.950000; // AR
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
  ru = 8.31446261815324e+07;
  ruc = 1.98721558317399615845;
  pa = 1.01325e+06;
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cv_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 7: HE
  species[7] = +1.50000000e+00;
  // species 8: AR
  species[8] = +1.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = +2.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
                 6.91588753e-08 * T3 + 2.69884373e-11 * T4;
    // species 1: H2
    species[1] = +1.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
                 2.01572094e-08 * T3 - 7.37611761e-12 * T4;
    // species 2: O2
    species[2] = +2.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
                 9.68129509e-09 * T3 + 3.24372837e-12 * T4;
    // species 3: H2O
    species[3] = +3.19863520e+00 - 2.03640170e-03 * T + 6.52034160e-06 * T2 -
                 5.48792690e-09 * T3 + 1.77196800e-12 * T4;
    // species 4: CO
    species[4] = +2.57953350e+00 - 6.10353690e-04 * T + 1.01681430e-06 * T2 +
                 9.07005860e-10 * T3 - 9.04424490e-13 * T4;
    // species 5: CO2
    species[5] = +1.35681300e+00 + 8.98412990e-03 * T - 7.12206320e-06 * T2 +
                 2.45730080e-09 * T3 - 1.42885480e-13 * T4;
    // species 6: N2
    species[6] = +2.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485500e-12 * T4;
  } else {
    // species 0: C2H4
    species[0] = +1.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
                 1.47222923e-09 * T3 - 1.25706061e-13 * T4;
    // species 1: H2
    species[1] = +1.93286575e+00 + 8.26608026e-04 * T - 1.46402364e-07 * T2 +
                 1.54100414e-11 * T3 - 6.88804800e-16 * T4;
    // species 2: O2
    species[2] = +2.66096065e+00 + 6.56365811e-04 * T - 1.41149627e-07 * T2 +
                 2.05797935e-11 * T3 - 1.29913436e-15 * T4;
    // species 3: H2O
    species[3] = +1.67703890e+00 + 2.97318160e-03 * T - 7.73768890e-07 * T2 +
                 9.44335140e-11 * T3 - 4.26899910e-15 * T4;
    // species 4: CO
    species[4] = +2.04848590e+00 + 1.35172810e-03 * T - 4.85794050e-07 * T2 +
                 7.88536440e-11 * T3 - 4.69807460e-15 * T4;
    // species 5: CO2
    species[5] = +3.63651110e+00 + 2.74145690e-03 * T - 9.95897590e-07 * T2 +
                 1.60386660e-10 * T3 - 9.16198570e-15 * T4;
    // species 6: N2
    species[6] = +1.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
                 1.00970400e-10 * T3 - 6.75335100e-15 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 7: HE
  species[7] = +2.50000000e+00;
  // species 8: AR
  species[8] = +2.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = +3.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
                 6.91588753e-08 * T3 + 2.69884373e-11 * T4;
    // species 1: H2
    species[1] = +2.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
                 2.01572094e-08 * T3 - 7.37611761e-12 * T4;
    // species 2: O2
    species[2] = +3.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
                 9.68129509e-09 * T3 + 3.24372837e-12 * T4;
    // species 3: H2O
    species[3] = +4.19863520e+00 - 2.03640170e-03 * T + 6.52034160e-06 * T2 -
                 5.48792690e-09 * T3 + 1.77196800e-12 * T4;
    // species 4: CO
    species[4] = +3.57953350e+00 - 6.10353690e-04 * T + 1.01681430e-06 * T2 +
                 9.07005860e-10 * T3 - 9.04424490e-13 * T4;
    // species 5: CO2
    species[5] = +2.35681300e+00 + 8.98412990e-03 * T - 7.12206320e-06 * T2 +
                 2.45730080e-09 * T3 - 1.42885480e-13 * T4;
    // species 6: N2
    species[6] = +3.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485500e-12 * T4;
  } else {
    // species 0: C2H4
    species[0] = +2.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
                 1.47222923e-09 * T3 - 1.25706061e-13 * T4;
    // species 1: H2
    species[1] = +2.93286575e+00 + 8.26608026e-04 * T - 1.46402364e-07 * T2 +
                 1.54100414e-11 * T3 - 6.88804800e-16 * T4;
    // species 2: O2
    species[2] = +3.66096065e+00 + 6.56365811e-04 * T - 1.41149627e-07 * T2 +
                 2.05797935e-11 * T3 - 1.29913436e-15 * T4;
    // species 3: H2O
    species[3] = +2.67703890e+00 + 2.97318160e-03 * T - 7.73768890e-07 * T2 +
                 9.44335140e-11 * T3 - 4.26899910e-15 * T4;
    // species 4: CO
    species[4] = +3.04848590e+00 + 1.35172810e-03 * T - 4.85794050e-07 * T2 +
                 7.88536440e-11 * T3 - 4.69807460e-15 * T4;
    // species 5: CO2
    species[5] = +4.63651110e+00 + 2.74145690e-03 * T - 9.95897590e-07 * T2 +
                 1.60386660e-10 * T3 - 9.16198570e-15 * T4;
    // species 6: N2
    species[6] = +2.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
                 1.00970400e-10 * T3 - 6.75335100e-15 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 7: HE
  species[7] = -7.453750000000000e+02 * invT + 1.571276026000000e+00 -
               2.500000000000000e+00 * logT;
  // species 8: AR
  species[8] = -7.453750000000000e+02 * invT - 1.866000000000000e+00 -
               2.500000000000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = +5.089775930000000e+03 * invT - 1.381294799999999e-01 -
                 3.959201480000000e+00 * logT + 3.785261235000000e-03 * T -
                 9.516504866666667e-06 * T2 + 5.763239608333333e-09 * T3 -
                 1.349421865000000e-12 * T4;
    // species 1: H2
    species[1] = -9.179351730000000e+02 * invT + 1.661320882000000e+00 -
                 2.344331120000000e+00 * logT - 3.990260375000000e-03 * T +
                 3.246358500000000e-06 * T2 - 1.679767450000000e-09 * T3 +
                 3.688058805000000e-13 * T4;
    // species 2: O2
    species[2] = -1.063943560000000e+03 * invT + 1.247806300000001e-01 -
                 3.782456360000000e+00 * logT + 1.498367080000000e-03 * T -
                 1.641217001666667e-06 * T2 + 8.067745908333334e-10 * T3 -
                 1.621864185000000e-13 * T4;
    // species 3: H2O
    species[3] = -3.029372600000000e+04 * invT + 5.047644210000000e+00 -
                 4.198635200000000e+00 * logT + 1.018200850000000e-03 * T -
                 1.086723600000000e-06 * T2 + 4.573272416666666e-10 * T3 -
                 8.859840000000001e-14 * T4;
    // species 4: CO
    species[4] = -1.434408600000000e+04 * invT + 7.112420000000030e-02 -
                 3.579533500000000e+00 * logT + 3.051768450000000e-04 * T -
                 1.694690500000000e-07 * T2 - 7.558382166666667e-11 * T3 +
                 4.522122450000000e-14 * T4;
    // species 5: CO2
    species[5] = -4.837197100000000e+04 * invT - 7.544090500000000e+00 -
                 2.356813000000000e+00 * logT - 4.492064950000000e-03 * T +
                 1.187010533333333e-06 * T2 - 2.047750666666667e-10 * T3 +
                 7.144273999999999e-15 * T4;
    // species 6: N2
    species[6] = -1.020900000000000e+03 * invT - 6.516950000000001e-01 -
                 3.298677000000000e+00 * logT - 7.041200000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427500000000e-13 * T4;
  } else {
    // species 0: C2H4
    species[0] = +4.939886140000000e+03 * invT - 8.269258140000002e+00 -
                 2.036111160000000e+00 * logT - 7.322707550000000e-03 * T +
                 1.118463191666667e-06 * T2 - 1.226857691666667e-10 * T3 +
                 6.285303050000000e-15 * T4;
    // species 1: H2
    species[1] = -8.130655810000000e+02 * invT + 3.957194400000000e+00 -
                 2.932865750000000e+00 * logT - 4.133040130000000e-04 * T +
                 2.440039400000000e-08 * T2 - 1.284170116666667e-12 * T3 +
                 3.444024000000000e-17 * T4;
    // species 2: O2
    species[2] = -1.215977180000000e+03 * invT + 2.455978599999997e-01 -
                 3.660960650000000e+00 * logT - 3.281829055000000e-04 * T +
                 2.352493783333333e-08 * T2 - 1.714982791666666e-12 * T3 +
                 6.495671800000001e-17 * T4;
    // species 3: H2O
    species[3] = -2.988589400000000e+04 * invT - 4.205511100000001e+00 -
                 2.677038900000000e+00 * logT - 1.486590800000000e-03 * T +
                 1.289614816666667e-07 * T2 - 7.869459500000001e-12 * T3 +
                 2.134499550000000e-16 * T4;
    // species 4: CO
    species[4] = -1.426611700000000e+04 * invT - 2.968611800000000e+00 -
                 3.048485900000000e+00 * logT - 6.758640500000000e-04 * T +
                 8.096567500000000e-08 * T2 - 6.571137000000000e-12 * T3 +
                 2.349037300000000e-16 * T4;
    // species 5: CO2
    species[5] = -4.902490400000000e+04 * invT + 6.571406600000000e+00 -
                 4.636511100000000e+00 * logT - 1.370728450000000e-03 * T +
                 1.659829316666667e-07 * T2 - 1.336555500000000e-11 * T3 +
                 4.580992850000000e-16 * T4;
    // species 6: N2
    species[6] = -9.227977000000000e+02 * invT - 3.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439885000000000e-04 * T +
                 9.474601666666666e-08 * T2 - 8.414199999999999e-12 * T3 +
                 3.376675500000000e-16 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 7: HE
  species[7] = -7.45375000e+02 * invT + 5.71276026e-01 - 2.50000000e+00 * logT;
  // species 8: AR
  species[8] = -7.45375000e+02 * invT - 2.86600000e+00 - 2.50000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = +5.08977593e+03 * invT - 1.13812948e+00 -
                 3.95920148e+00 * logT + 3.78526124e-03 * T -
                 9.51650487e-06 * T2 + 5.76323961e-09 * T3 -
                 1.34942187e-12 * T4;
    // species 1: H2
    species[1] = -9.17935173e+02 * invT + 6.61320882e-01 -
                 2.34433112e+00 * logT - 3.99026037e-03 * T +
                 3.24635850e-06 * T2 - 1.67976745e-09 * T3 +
                 3.68805881e-13 * T4;
    // species 2: O2
    species[2] = -1.06394356e+03 * invT - 8.75219370e-01 -
                 3.78245636e+00 * logT + 1.49836708e-03 * T -
                 1.64121700e-06 * T2 + 8.06774591e-10 * T3 -
                 1.62186418e-13 * T4;
    // species 3: H2O
    species[3] = -3.02937260e+04 * invT + 4.04764421e+00 -
                 4.19863520e+00 * logT + 1.01820085e-03 * T -
                 1.08672360e-06 * T2 + 4.57327242e-10 * T3 -
                 8.85984000e-14 * T4;
    // species 4: CO
    species[4] = -1.43440860e+04 * invT - 9.28875800e-01 -
                 3.57953350e+00 * logT + 3.05176845e-04 * T -
                 1.69469050e-07 * T2 - 7.55838217e-11 * T3 +
                 4.52212245e-14 * T4;
    // species 5: CO2
    species[5] = -4.83719710e+04 * invT - 8.54409050e+00 -
                 2.35681300e+00 * logT - 4.49206495e-03 * T +
                 1.18701053e-06 * T2 - 2.04775067e-10 * T3 +
                 7.14427400e-15 * T4;
    // species 6: N2
    species[6] = -1.02090000e+03 * invT - 1.65169500e+00 -
                 3.29867700e+00 * logT - 7.04120000e-04 * T +
                 6.60537000e-07 * T2 - 4.70126250e-10 * T3 +
                 1.22242750e-13 * T4;
  } else {
    // species 0: C2H4
    species[0] = +4.93988614e+03 * invT - 9.26925814e+00 -
                 2.03611116e+00 * logT - 7.32270755e-03 * T +
                 1.11846319e-06 * T2 - 1.22685769e-10 * T3 +
                 6.28530305e-15 * T4;
    // species 1: H2
    species[1] = -8.13065581e+02 * invT + 2.95719440e+00 -
                 2.93286575e+00 * logT - 4.13304013e-04 * T +
                 2.44003940e-08 * T2 - 1.28417012e-12 * T3 +
                 3.44402400e-17 * T4;
    // species 2: O2
    species[2] = -1.21597718e+03 * invT - 7.54402140e-01 -
                 3.66096065e+00 * logT - 3.28182906e-04 * T +
                 2.35249378e-08 * T2 - 1.71498279e-12 * T3 +
                 6.49567180e-17 * T4;
    // species 3: H2O
    species[3] = -2.98858940e+04 * invT - 5.20551110e+00 -
                 2.67703890e+00 * logT - 1.48659080e-03 * T +
                 1.28961482e-07 * T2 - 7.86945950e-12 * T3 +
                 2.13449955e-16 * T4;
    // species 4: CO
    species[4] = -1.42661170e+04 * invT - 3.96861180e+00 -
                 3.04848590e+00 * logT - 6.75864050e-04 * T +
                 8.09656750e-08 * T2 - 6.57113700e-12 * T3 +
                 2.34903730e-16 * T4;
    // species 5: CO2
    species[5] = -4.90249040e+04 * invT + 5.57140660e+00 -
                 4.63651110e+00 * logT - 1.37072845e-03 * T +
                 1.65982932e-07 * T2 - 1.33655550e-11 * T3 +
                 4.58099285e-16 * T4;
    // species 6: N2
    species[6] = -9.22797700e+02 * invT - 4.05388800e+00 -
                 2.92664000e+00 * logT - 7.43988500e-04 * T +
                 9.47460167e-08 * T2 - 8.41420000e-12 * T3 +
                 3.37667550e-16 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 7: HE
  species[7] = +1.50000000e+00 - 7.45375000e+02 * invT;
  // species 8: AR
  species[8] = +1.50000000e+00 - 7.45375000e+02 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = +2.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
                 1.72897188e-08 * T3 + 5.39768746e-12 * T4 +
                 5.08977593e+03 * invT;
    // species 1: H2
    species[1] = +1.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
                 5.03930235e-09 * T3 - 1.47522352e-12 * T4 -
                 9.17935173e+02 * invT;
    // species 2: O2
    species[2] = +2.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
                 2.42032377e-09 * T3 + 6.48745674e-13 * T4 -
                 1.06394356e+03 * invT;
    // species 3: H2O
    species[3] = +3.19863520e+00 - 1.01820085e-03 * T + 2.17344720e-06 * T2 -
                 1.37198172e-09 * T3 + 3.54393600e-13 * T4 -
                 3.02937260e+04 * invT;
    // species 4: CO
    species[4] = +2.57953350e+00 - 3.05176845e-04 * T + 3.38938100e-07 * T2 +
                 2.26751465e-10 * T3 - 1.80884898e-13 * T4 -
                 1.43440860e+04 * invT;
    // species 5: CO2
    species[5] = +1.35681300e+00 + 4.49206495e-03 * T - 2.37402107e-06 * T2 +
                 6.14325200e-10 * T3 - 2.85770960e-14 * T4 -
                 4.83719710e+04 * invT;
    // species 6: N2
    species[6] = +2.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88971000e-13 * T4 -
                 1.02090000e+03 * invT;
  } else {
    // species 0: C2H4
    species[0] = +1.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
                 3.68057308e-10 * T3 - 2.51412122e-14 * T4 +
                 4.93988614e+03 * invT;
    // species 1: H2
    species[1] = +1.93286575e+00 + 4.13304013e-04 * T - 4.88007880e-08 * T2 +
                 3.85251035e-12 * T3 - 1.37760960e-16 * T4 -
                 8.13065581e+02 * invT;
    // species 2: O2
    species[2] = +2.66096065e+00 + 3.28182906e-04 * T - 4.70498757e-08 * T2 +
                 5.14494837e-12 * T3 - 2.59826872e-16 * T4 -
                 1.21597718e+03 * invT;
    // species 3: H2O
    species[3] = +1.67703890e+00 + 1.48659080e-03 * T - 2.57922963e-07 * T2 +
                 2.36083785e-11 * T3 - 8.53799820e-16 * T4 -
                 2.98858940e+04 * invT;
    // species 4: CO
    species[4] = +2.04848590e+00 + 6.75864050e-04 * T - 1.61931350e-07 * T2 +
                 1.97134110e-11 * T3 - 9.39614920e-16 * T4 -
                 1.42661170e+04 * invT;
    // species 5: CO2
    species[5] = +3.63651110e+00 + 1.37072845e-03 * T - 3.31965863e-07 * T2 +
                 4.00966650e-11 * T3 - 1.83239714e-15 * T4 -
                 4.90249040e+04 * invT;
    // species 6: N2
    species[6] = +1.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
                 2.52426000e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 7: HE
  species[7] = +2.50000000e+00 - 7.45375000e+02 * invT;
  // species 8: AR
  species[8] = +2.50000000e+00 - 7.45375000e+02 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = +3.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
                 1.72897188e-08 * T3 + 5.39768746e-12 * T4 +
                 5.08977593e+03 * invT;
    // species 1: H2
    species[1] = +2.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
                 5.03930235e-09 * T3 - 1.47522352e-12 * T4 -
                 9.17935173e+02 * invT;
    // species 2: O2
    species[2] = +3.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
                 2.42032377e-09 * T3 + 6.48745674e-13 * T4 -
                 1.06394356e+03 * invT;
    // species 3: H2O
    species[3] = +4.19863520e+00 - 1.01820085e-03 * T + 2.17344720e-06 * T2 -
                 1.37198172e-09 * T3 + 3.54393600e-13 * T4 -
                 3.02937260e+04 * invT;
    // species 4: CO
    species[4] = +3.57953350e+00 - 3.05176845e-04 * T + 3.38938100e-07 * T2 +
                 2.26751465e-10 * T3 - 1.80884898e-13 * T4 -
                 1.43440860e+04 * invT;
    // species 5: CO2
    species[5] = +2.35681300e+00 + 4.49206495e-03 * T - 2.37402107e-06 * T2 +
                 6.14325200e-10 * T3 - 2.85770960e-14 * T4 -
                 4.83719710e+04 * invT;
    // species 6: N2
    species[6] = +3.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88971000e-13 * T4 -
                 1.02090000e+03 * invT;
  } else {
    // species 0: C2H4
    species[0] = +2.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
                 3.68057308e-10 * T3 - 2.51412122e-14 * T4 +
                 4.93988614e+03 * invT;
    // species 1: H2
    species[1] = +2.93286575e+00 + 4.13304013e-04 * T - 4.88007880e-08 * T2 +
                 3.85251035e-12 * T3 - 1.37760960e-16 * T4 -
                 8.13065581e+02 * invT;
    // species 2: O2
    species[2] = +3.66096065e+00 + 3.28182906e-04 * T - 4.70498757e-08 * T2 +
                 5.14494837e-12 * T3 - 2.59826872e-16 * T4 -
                 1.21597718e+03 * invT;
    // species 3: H2O
    species[3] = +2.67703890e+00 + 1.48659080e-03 * T - 2.57922963e-07 * T2 +
                 2.36083785e-11 * T3 - 8.53799820e-16 * T4 -
                 2.98858940e+04 * invT;
    // species 4: CO
    species[4] = +3.04848590e+00 + 6.75864050e-04 * T - 1.61931350e-07 * T2 +
                 1.97134110e-11 * T3 - 9.39614920e-16 * T4 -
                 1.42661170e+04 * invT;
    // species 5: CO2
    species[5] = +4.63651110e+00 + 1.37072845e-03 * T - 3.31965863e-07 * T2 +
                 4.00966650e-11 * T3 - 1.83239714e-15 * T4 -
                 4.90249040e+04 * invT;
    // species 6: N2
    species[6] = +2.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
                 2.52426000e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 7: HE
  species[7] = +2.50000000e+00 * logT + 9.28723974e-01;
  // species 8: AR
  species[8] = +2.50000000e+00 * logT + 4.36600000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = +3.95920148e+00 * logT - 7.57052247e-03 * T +
                 2.85495146e-05 * T2 - 2.30529584e-08 * T3 +
                 6.74710933e-12 * T4 + 4.09733096e+00;
    // species 1: H2
    species[1] = +2.34433112e+00 * logT + 7.98052075e-03 * T -
                 9.73907550e-06 * T2 + 6.71906980e-09 * T3 -
                 1.84402940e-12 * T4 + 6.83010238e-01;
    // species 2: O2
    species[2] = +3.78245636e+00 * logT - 2.99673416e-03 * T +
                 4.92365101e-06 * T2 - 3.22709836e-09 * T3 +
                 8.10932092e-13 * T4 + 3.65767573e+00;
    // species 3: H2O
    species[3] = +4.19863520e+00 * logT - 2.03640170e-03 * T +
                 3.26017080e-06 * T2 - 1.82930897e-09 * T3 +
                 4.42992000e-13 * T4 - 8.49009010e-01;
    // species 4: CO
    species[4] = +3.57953350e+00 * logT - 6.10353690e-04 * T +
                 5.08407150e-07 * T2 + 3.02335287e-10 * T3 -
                 2.26106122e-13 * T4 + 3.50840930e+00;
    // species 5: CO2
    species[5] = +2.35681300e+00 * logT + 8.98412990e-03 * T -
                 3.56103160e-06 * T2 + 8.19100267e-10 * T3 -
                 3.57213700e-14 * T4 + 9.90090350e+00;
    // species 6: N2
    species[6] = +3.29867700e+00 * logT + 1.40824000e-03 * T -
                 1.98161100e-06 * T2 + 1.88050500e-09 * T3 -
                 6.11213750e-13 * T4 + 3.95037200e+00;
  } else {
    // species 0: C2H4
    species[0] = +2.03611116e+00 * logT + 1.46454151e-02 * T -
                 3.35538958e-06 * T2 + 4.90743077e-10 * T3 -
                 3.14265152e-14 * T4 + 1.03053693e+01;
    // species 1: H2
    species[1] = +2.93286575e+00 * logT + 8.26608026e-04 * T -
                 7.32011820e-08 * T2 + 5.13668047e-12 * T3 -
                 1.72201200e-16 * T4 - 1.02432865e+00;
    // species 2: O2
    species[2] = +3.66096065e+00 * logT + 6.56365811e-04 * T -
                 7.05748135e-08 * T2 + 6.85993117e-12 * T3 -
                 3.24783590e-16 * T4 + 3.41536279e+00;
    // species 3: H2O
    species[3] = +2.67703890e+00 * logT + 2.97318160e-03 * T -
                 3.86884445e-07 * T2 + 3.14778380e-11 * T3 -
                 1.06724977e-15 * T4 + 6.88255000e+00;
    // species 4: CO
    species[4] = +3.04848590e+00 * logT + 1.35172810e-03 * T -
                 2.42897025e-07 * T2 + 2.62845480e-11 * T3 -
                 1.17451865e-15 * T4 + 6.01709770e+00;
    // species 5: CO2
    species[5] = +4.63651110e+00 * logT + 2.74145690e-03 * T -
                 4.97948795e-07 * T2 + 5.34622200e-11 * T3 -
                 2.29049642e-15 * T4 - 1.93489550e+00;
    // species 6: N2
    species[6] = +2.92664000e+00 * logT + 1.48797700e-03 * T -
                 2.84238050e-07 * T2 + 3.36568000e-11 * T3 -
                 1.68833775e-15 * T4 + 5.98052800e+00;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;

  // species with no change across T
  // species 7: HE
  species[7] = 0.0;
  // species 8: AR
  species[8] = 0.0;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    species[0] = -7.57052247e-03 + 1.14198058e-04 * T - 2.07476626e-07 * T2 +
                 1.07953749e-10 * T3;
    // species 1: H2
    species[1] = +7.98052075e-03 - 3.89563020e-05 * T + 6.04716282e-08 * T2 -
                 2.95044704e-11 * T3;
    // species 2: O2
    species[2] = -2.99673416e-03 + 1.96946040e-05 * T - 2.90438853e-08 * T2 +
                 1.29749135e-11 * T3;
    // species 3: H2O
    species[3] = -2.03640170e-03 + 1.30406832e-05 * T - 1.64637807e-08 * T2 +
                 7.08787200e-12 * T3;
    // species 4: CO
    species[4] = -6.10353690e-04 + 2.03362860e-06 * T + 2.72101758e-09 * T2 -
                 3.61769796e-12 * T3;
    // species 5: CO2
    species[5] = +8.98412990e-03 - 1.42441264e-05 * T + 7.37190240e-09 * T2 -
                 5.71541920e-13 * T3;
    // species 6: N2
    species[6] = +1.40824000e-03 - 7.92644400e-06 * T + 1.69245450e-08 * T2 -
                 9.77942000e-12 * T3;
  } else {
    // species 0: C2H4
    species[0] = +1.46454151e-02 - 1.34215583e-05 * T + 4.41668769e-09 * T2 -
                 5.02824244e-13 * T3;
    // species 1: H2
    species[1] = +8.26608026e-04 - 2.92804728e-07 * T + 4.62301242e-11 * T2 -
                 2.75521920e-15 * T3;
    // species 2: O2
    species[2] = +6.56365811e-04 - 2.82299254e-07 * T + 6.17393805e-11 * T2 -
                 5.19653744e-15 * T3;
    // species 3: H2O
    species[3] = +2.97318160e-03 - 1.54753778e-06 * T + 2.83300542e-10 * T2 -
                 1.70759964e-14 * T3;
    // species 4: CO
    species[4] = +1.35172810e-03 - 9.71588100e-07 * T + 2.36560932e-10 * T2 -
                 1.87922984e-14 * T3;
    // species 5: CO2
    species[5] = +2.74145690e-03 - 1.99179518e-06 * T + 4.81159980e-10 * T2 -
                 3.66479428e-14 * T3;
    // species 6: N2
    species[6] = +1.48797700e-03 - 1.13695220e-06 * T + 3.02911200e-10 * T2 -
                 2.70134040e-14 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[9]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * cpor[id];
  }

  cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
  amrex::Real result = 0.0;

  // compute Cp/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 7: HE
  result += y[7] * (+2.50000000e+00) * 0.2498374807187924;
  // species 8: AR
  result += y[8] * (+2.50000000e+00) * 0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    result += y[0] *
              (+3.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
               6.91588753e-08 * T3 + 2.69884373e-11 * T4) *
              0.0356455407428531;
    // species 1: H2
    result += y[1] *
              (+2.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
               2.01572094e-08 * T3 - 7.37611761e-12 * T4) *
              0.4960317460317460;
    // species 2: O2
    result += y[2] *
              (+3.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
               9.68129509e-09 * T3 + 3.24372837e-12 * T4) *
              0.0312519532470779;
    // species 3: H2O
    result += y[3] *
              (+4.19863520e+00 - 2.03640170e-03 * T + 6.52034160e-06 * T2 -
               5.48792690e-09 * T3 + 1.77196800e-12 * T4) *
              0.0555092978073827;
    // species 4: CO
    result += y[4] *
              (+3.57953350e+00 - 6.10353690e-04 * T + 1.01681430e-06 * T2 +
               9.07005860e-10 * T3 - 9.04424490e-13 * T4) *
              0.0357015351660121;
    // species 5: CO2
    result += y[5] *
              (+2.35681300e+00 + 8.98412990e-03 * T - 7.12206320e-06 * T2 +
               2.45730080e-09 * T3 - 1.42885480e-13 * T4) *
              0.0227226249176305;
    // species 6: N2
    result += y[6] *
              (+3.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485500e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 0: C2H4
    result += y[0] *
              (+2.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
               1.47222923e-09 * T3 - 1.25706061e-13 * T4) *
              0.0356455407428531;
    // species 1: H2
    result += y[1] *
              (+2.93286575e+00 + 8.26608026e-04 * T - 1.46402364e-07 * T2 +
               1.54100414e-11 * T3 - 6.88804800e-16 * T4) *
              0.4960317460317460;
    // species 2: O2
    result += y[2] *
              (+3.66096065e+00 + 6.56365811e-04 * T - 1.41149627e-07 * T2 +
               2.05797935e-11 * T3 - 1.29913436e-15 * T4) *
              0.0312519532470779;
    // species 3: H2O
    result += y[3] *
              (+2.67703890e+00 + 2.97318160e-03 * T - 7.73768890e-07 * T2 +
               9.44335140e-11 * T3 - 4.26899910e-15 * T4) *
              0.0555092978073827;
    // species 4: CO
    result += y[4] *
              (+3.04848590e+00 + 1.35172810e-03 * T - 4.85794050e-07 * T2 +
               7.88536440e-11 * T3 - 4.69807460e-15 * T4) *
              0.0357015351660121;
    // species 5: CO2
    result += y[5] *
              (+4.63651110e+00 + 2.74145690e-03 * T - 9.95897590e-07 * T2 +
               1.60386660e-10 * T3 - 9.16198570e-15 * T4) *
              0.0227226249176305;
    // species 6: N2
    result += y[6] *
              (+2.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
               1.00970400e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[9]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * cvor[id];
  }

  cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cvbs)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 7: HE
  result += y[7] * (+1.50000000e+00) * 0.2498374807187924;
  // species 8: AR
  result += y[8] * (+1.50000000e+00) * 0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    result += y[0] *
              (+2.95920148e+00 - 7.57052247e-03 * T + 5.70990292e-05 * T2 -
               6.91588753e-08 * T3 + 2.69884373e-11 * T4) *
              0.0356455407428531;
    // species 1: H2
    result += y[1] *
              (+1.34433112e+00 + 7.98052075e-03 * T - 1.94781510e-05 * T2 +
               2.01572094e-08 * T3 - 7.37611761e-12 * T4) *
              0.4960317460317460;
    // species 2: O2
    result += y[2] *
              (+2.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
               9.68129509e-09 * T3 + 3.24372837e-12 * T4) *
              0.0312519532470779;
    // species 3: H2O
    result += y[3] *
              (+3.19863520e+00 - 2.03640170e-03 * T + 6.52034160e-06 * T2 -
               5.48792690e-09 * T3 + 1.77196800e-12 * T4) *
              0.0555092978073827;
    // species 4: CO
    result += y[4] *
              (+2.57953350e+00 - 6.10353690e-04 * T + 1.01681430e-06 * T2 +
               9.07005860e-10 * T3 - 9.04424490e-13 * T4) *
              0.0357015351660121;
    // species 5: CO2
    result += y[5] *
              (+1.35681300e+00 + 8.98412990e-03 * T - 7.12206320e-06 * T2 +
               2.45730080e-09 * T3 - 1.42885480e-13 * T4) *
              0.0227226249176305;
    // species 6: N2
    result += y[6] *
              (+2.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485500e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 0: C2H4
    result += y[0] *
              (+1.03611116e+00 + 1.46454151e-02 * T - 6.71077915e-06 * T2 +
               1.47222923e-09 * T3 - 1.25706061e-13 * T4) *
              0.0356455407428531;
    // species 1: H2
    result += y[1] *
              (+1.93286575e+00 + 8.26608026e-04 * T - 1.46402364e-07 * T2 +
               1.54100414e-11 * T3 - 6.88804800e-16 * T4) *
              0.4960317460317460;
    // species 2: O2
    result += y[2] *
              (+2.66096065e+00 + 6.56365811e-04 * T - 1.41149627e-07 * T2 +
               2.05797935e-11 * T3 - 1.29913436e-15 * T4) *
              0.0312519532470779;
    // species 3: H2O
    result += y[3] *
              (+1.67703890e+00 + 2.97318160e-03 * T - 7.73768890e-07 * T2 +
               9.44335140e-11 * T3 - 4.26899910e-15 * T4) *
              0.0555092978073827;
    // species 4: CO
    result += y[4] *
              (+2.04848590e+00 + 1.35172810e-03 * T - 4.85794050e-07 * T2 +
               7.88536440e-11 * T3 - 4.69807460e-15 * T4) *
              0.0357015351660121;
    // species 5: CO2
    result += y[5] *
              (+3.63651110e+00 + 2.74145690e-03 * T - 9.95897590e-07 * T2 +
               1.60386660e-10 * T3 - 9.16198570e-15 * T4) *
              0.0227226249176305;
    // species 6: N2
    result += y[6] *
              (+1.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
               1.00970400e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[9];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * hml[id];
  }

  hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& hbms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 7: HE
  result +=
    y[7] * (+2.50000000e+00 - 7.45375000e+02 * invT) * 0.2498374807187924;
  // species 8: AR
  result +=
    y[8] * (+2.50000000e+00 - 7.45375000e+02 * invT) * 0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    result +=
      y[0] *
      (+3.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
       1.72897188e-08 * T3 + 5.39768746e-12 * T4 + 5.08977593e+03 * invT) *
      0.0356455407428531;
    // species 1: H2
    result +=
      y[1] *
      (+2.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
       5.03930235e-09 * T3 - 1.47522352e-12 * T4 - 9.17935173e+02 * invT) *
      0.4960317460317460;
    // species 2: O2
    result +=
      y[2] *
      (+3.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
       2.42032377e-09 * T3 + 6.48745674e-13 * T4 - 1.06394356e+03 * invT) *
      0.0312519532470779;
    // species 3: H2O
    result +=
      y[3] *
      (+4.19863520e+00 - 1.01820085e-03 * T + 2.17344720e-06 * T2 -
       1.37198172e-09 * T3 + 3.54393600e-13 * T4 - 3.02937260e+04 * invT) *
      0.0555092978073827;
    // species 4: CO
    result +=
      y[4] *
      (+3.57953350e+00 - 3.05176845e-04 * T + 3.38938100e-07 * T2 +
       2.26751465e-10 * T3 - 1.80884898e-13 * T4 - 1.43440860e+04 * invT) *
      0.0357015351660121;
    // species 5: CO2
    result +=
      y[5] *
      (+2.35681300e+00 + 4.49206495e-03 * T - 2.37402107e-06 * T2 +
       6.14325200e-10 * T3 - 2.85770960e-14 * T4 - 4.83719710e+04 * invT) *
      0.0227226249176305;
    // species 6: N2
    result +=
      y[6] *
      (+3.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88971000e-13 * T4 - 1.02090000e+03 * invT) *
      0.0356964374955379;
  } else {
    // species 0: C2H4
    result +=
      y[0] *
      (+2.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
       3.68057308e-10 * T3 - 2.51412122e-14 * T4 + 4.93988614e+03 * invT) *
      0.0356455407428531;
    // species 1: H2
    result +=
      y[1] *
      (+2.93286575e+00 + 4.13304013e-04 * T - 4.88007880e-08 * T2 +
       3.85251035e-12 * T3 - 1.37760960e-16 * T4 - 8.13065581e+02 * invT) *
      0.4960317460317460;
    // species 2: O2
    result +=
      y[2] *
      (+3.66096065e+00 + 3.28182906e-04 * T - 4.70498757e-08 * T2 +
       5.14494837e-12 * T3 - 2.59826872e-16 * T4 - 1.21597718e+03 * invT) *
      0.0312519532470779;
    // species 3: H2O
    result +=
      y[3] *
      (+2.67703890e+00 + 1.48659080e-03 * T - 2.57922963e-07 * T2 +
       2.36083785e-11 * T3 - 8.53799820e-16 * T4 - 2.98858940e+04 * invT) *
      0.0555092978073827;
    // species 4: CO
    result +=
      y[4] *
      (+3.04848590e+00 + 6.75864050e-04 * T - 1.61931350e-07 * T2 +
       1.97134110e-11 * T3 - 9.39614920e-16 * T4 - 1.42661170e+04 * invT) *
      0.0357015351660121;
    // species 5: CO2
    result +=
      y[5] *
      (+4.63651110e+00 + 1.37072845e-03 * T - 3.31965863e-07 * T2 +
       4.00966650e-11 * T3 - 1.83239714e-15 * T4 - 4.90249040e+04 * invT) *
      0.0227226249176305;
    // species 6: N2
    result +=
      y[6] *
      (+2.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
       2.52426000e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[9];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 9; ++id) {
    result += x[id] * uml[id];
  }

  ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 7: HE
  result +=
    y[7] * (+1.50000000e+00 - 7.45375000e+02 * invT) * 0.2498374807187924;
  // species 8: AR
  result +=
    y[8] * (+1.50000000e+00 - 7.45375000e+02 * invT) * 0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: C2H4
    result +=
      y[0] *
      (+2.95920148e+00 - 3.78526124e-03 * T + 1.90330097e-05 * T2 -
       1.72897188e-08 * T3 + 5.39768746e-12 * T4 + 5.08977593e+03 * invT) *
      0.0356455407428531;
    // species 1: H2
    result +=
      y[1] *
      (+1.34433112e+00 + 3.99026037e-03 * T - 6.49271700e-06 * T2 +
       5.03930235e-09 * T3 - 1.47522352e-12 * T4 - 9.17935173e+02 * invT) *
      0.4960317460317460;
    // species 2: O2
    result +=
      y[2] *
      (+2.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
       2.42032377e-09 * T3 + 6.48745674e-13 * T4 - 1.06394356e+03 * invT) *
      0.0312519532470779;
    // species 3: H2O
    result +=
      y[3] *
      (+3.19863520e+00 - 1.01820085e-03 * T + 2.17344720e-06 * T2 -
       1.37198172e-09 * T3 + 3.54393600e-13 * T4 - 3.02937260e+04 * invT) *
      0.0555092978073827;
    // species 4: CO
    result +=
      y[4] *
      (+2.57953350e+00 - 3.05176845e-04 * T + 3.38938100e-07 * T2 +
       2.26751465e-10 * T3 - 1.80884898e-13 * T4 - 1.43440860e+04 * invT) *
      0.0357015351660121;
    // species 5: CO2
    result +=
      y[5] *
      (+1.35681300e+00 + 4.49206495e-03 * T - 2.37402107e-06 * T2 +
       6.14325200e-10 * T3 - 2.85770960e-14 * T4 - 4.83719710e+04 * invT) *
      0.0227226249176305;
    // species 6: N2
    result +=
      y[6] *
      (+2.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88971000e-13 * T4 - 1.02090000e+03 * invT) *
      0.0356964374955379;
  } else {
    // species 0: C2H4
    result +=
      y[0] *
      (+1.03611116e+00 + 7.32270755e-03 * T - 2.23692638e-06 * T2 +
       3.68057308e-10 * T3 - 2.51412122e-14 * T4 + 4.93988614e+03 * invT) *
      0.0356455407428531;
    // species 1: H2
    result +=
      y[1] *
      (+1.93286575e+00 + 4.13304013e-04 * T - 4.88007880e-08 * T2 +
       3.85251035e-12 * T3 - 1.37760960e-16 * T4 - 8.13065581e+02 * invT) *
      0.4960317460317460;
    // species 2: O2
    result +=
      y[2] *
      (+2.66096065e+00 + 3.28182906e-04 * T - 4.70498757e-08 * T2 +
       5.14494837e-12 * T3 - 2.59826872e-16 * T4 - 1.21597718e+03 * invT) *
      0.0312519532470779;
    // species 3: H2O
    result +=
      y[3] *
      (+1.67703890e+00 + 1.48659080e-03 * T - 2.57922963e-07 * T2 +
       2.36083785e-11 * T3 - 8.53799820e-16 * T4 - 2.98858940e+04 * invT) *
      0.0555092978073827;
    // species 4: CO
    result +=
      y[4] *
      (+2.04848590e+00 + 6.75864050e-04 * T - 1.61931350e-07 * T2 +
       1.97134110e-11 * T3 - 9.39614920e-16 * T4 - 1.42661170e+04 * invT) *
      0.0357015351660121;
    // species 5: CO2
    result +=
      y[5] *
      (+3.63651110e+00 + 1.37072845e-03 * T - 3.31965863e-07 * T2 +
       4.00966650e-11 * T3 - 1.83239714e-15 * T4 - 4.90249040e+04 * invT) *
      0.0227226249176305;
    // species 6: N2
    result +=
      y[6] *
      (+1.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
       2.52426000e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBML(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& sbml)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[9]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 9; ++id) {
    result += x[id] * (sor[id] - log((x[id] + 1e-100)) - logPratio);
  }

  sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBMS(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& sbms)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[9];  //  temporary storage
  amrex::Real x[9];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (28.054000 * YOW);
  x[1] = y[1] / (2.016000 * YOW);
  x[2] = y[2] / (31.998000 * YOW);
  x[3] = y[3] / (18.015000 * YOW);
  x[4] = y[4] / (28.010000 * YOW);
  x[5] = y[5] / (44.009000 * YOW);
  x[6] = y[6] / (28.014000 * YOW);
  x[7] = y[7] / (4.002602 * YOW);
  x[8] = y[8] / (39.950000 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 9; i++) {
    result += x[i] * (sor[i] - log((x[i] + 1e-100)) - logPratio);
  }
  // Scale by R/W
  sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_EY(
  const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin);
  CKUBMS(tmax, y, emax);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmax, y, cv);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(t1, y, e1);
    CKCVBS(t1, y, cv);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_HY(
  const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real h1, hmin, hmax, cp, t1, dt;
  CKHBMS(tmin, y, hmin);
  CKHBMS(tmax, y, hmax);
  if (h < hmin) {
    // Linear Extrapolation below tmin
    CKCPBS(tmin, y, cp);
    t = tmin - (hmin - h) / cp;
    ierr = 1;
    return;
  }
  if (h > hmax) {
    // Linear Extrapolation above tmax
    CKCPBS(tmax, y, cp);
    t = tmax - (hmax - h) / cp;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (hmax - hmin) * (h - hmin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKHBMS(t1, y, h1);
    CKCPBS(t1, y, cp);
    dt = (h - h1) / cp;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPX(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& P)
{
  amrex::Real XW = 0;                      //  To hold mean molecular wt
  XW += x[0] * 28.054000;                  // C2H4
  XW += x[1] * 2.016000;                   // H2
  XW += x[2] * 31.998000;                  // O2
  XW += x[3] * 18.015000;                  // H2O
  XW += x[4] * 28.010000;                  // CO
  XW += x[5] * 44.009000;                  // CO2
  XW += x[6] * 28.014000;                  // N2
  XW += x[7] * 4.002602;                   // HE
  XW += x[8] * 39.950000;                  // AR
  P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPY(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& P)
{
  amrex::Real YOW = 0; //  for computing mean MW

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }
  P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPC(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& P)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 28.054000; // C2H4
  W += c[1] * 2.016000;  // H2
  W += c[2] * 31.998000; // O2
  W += c[3] * 18.015000; // H2O
  W += c[4] * 28.010000; // CO
  W += c[5] * 44.009000; // CO2
  W += c[6] * 28.014000; // N2
  W += c[7] * 4.002602;  // HE
  W += c[8] * 39.950000; // AR

  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }
  P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W
}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOX(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& rho)
{
  amrex::Real XW = 0;                        //  To hold mean molecular wt
  XW += x[0] * 28.054000;                    // C2H4
  XW += x[1] * 2.016000;                     // H2
  XW += x[2] * 31.998000;                    // O2
  XW += x[3] * 18.015000;                    // H2O
  XW += x[4] * 28.010000;                    // CO
  XW += x[5] * 44.009000;                    // CO2
  XW += x[6] * 28.014000;                    // N2
  XW += x[7] * 4.002602;                     // HE
  XW += x[8] * 39.950000;                    // AR
  rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)
}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOY(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& rho)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }

  rho = P / (8.31446261815324e+07 * T * YOW); // rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOC(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& rho)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 28.054000; // C2H4
  W += c[1] * 2.016000;  // H2
  W += c[2] * 31.998000; // O2
  W += c[3] * 18.015000; // H2O
  W += c[4] * 28.010000; // CO
  W += c[5] * 44.009000; // CO2
  W += c[6] * 28.014000; // N2
  W += c[7] * 4.002602;  // HE
  W += c[8] * 39.950000; // AR

  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }
  rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWT(amrex::Real wt[])
{
  get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;     //  see Eq 4 in CK Manual
  XW += x[0] * 28.054000; // C2H4
  XW += x[1] * 2.016000;  // H2
  XW += x[2] * 31.998000; // O2
  XW += x[3] * 18.015000; // H2O
  XW += x[4] * 28.010000; // CO
  XW += x[5] * 44.009000; // CO2
  XW += x[6] * 28.014000; // N2
  XW += x[7] * 4.002602;  // HE
  XW += x[8] * 39.950000; // AR
  wtm = XW;
}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWC(const amrex::Real c[], amrex::Real& wtm)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 28.054000; // C2H4
  W += c[1] * 2.016000;  // H2
  W += c[2] * 31.998000; // O2
  W += c[3] * 18.015000; // H2O
  W += c[4] * 28.010000; // CO
  W += c[5] * 44.009000; // CO2
  W += c[6] * 28.014000; // N2
  W += c[7] * 4.002602;  // HE
  W += c[8] * 39.950000; // AR

  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }
  //  CK provides no guard against division by zero
  wtm = W / sumC;
}

// get Cp/R as a function of T
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
  cp_R(cpor, T);
}

// get H/RT as a function of T
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHORT(const amrex::Real T, amrex::Real hort[])
{
  speciesEnthalpy(hort, T);
}

// get S/R as a function of T
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSOR(const amrex::Real T, amrex::Real sor[])
{
  speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTX(const amrex::Real y[], amrex::Real x[])
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 9; i++) {
    x[i] = y[i] * imw(i) * YOWINV;
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real c[])
{
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 9; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 9; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 9; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCR(
  const amrex::Real rho,
  amrex::Real /*T*/,
  const amrex::Real y[],
  amrex::Real c[])
{

  for (int i = 0; i < 9; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 28.054000; // C2H4
  XW += x[1] * 2.016000;  // H2
  XW += x[2] * 31.998000; // O2
  XW += x[3] * 18.015000; // H2O
  XW += x[4] * 28.010000; // CO
  XW += x[5] * 44.009000; // CO2
  XW += x[6] * 28.014000; // N2
  XW += x[7] * 4.002602;  // HE
  XW += x[8] * 39.950000; // AR
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 28.054000 * XWinv;
  y[1] = x[1] * 2.016000 * XWinv;
  y[2] = x[2] * 31.998000 * XWinv;
  y[3] = x[3] * 18.015000 * XWinv;
  y[4] = x[4] * 28.010000 * XWinv;
  y[5] = x[5] * 44.009000 * XWinv;
  y[6] = x[6] * 28.014000 * XWinv;
  y[7] = x[7] * 4.002602 * XWinv;
  y[8] = x[8] * 39.950000 * XWinv;
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real PORT = P / (8.31446261815324e+07 * T); // P/RT

  // Compute conversion, see Eq 10
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * PORT;
  }
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCR(
  const amrex::Real rho,
  const amrex::Real /*T*/,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 28.054000; // C2H4
  XW += x[1] * 2.016000;  // H2
  XW += x[2] * 31.998000; // O2
  XW += x[3] * 18.015000; // H2O
  XW += x[4] * 28.010000; // CO
  XW += x[5] * 44.009000; // CO2
  XW += x[6] * 28.014000; // N2
  XW += x[7] * 4.002602;  // HE
  XW += x[8] * 39.950000; // AR
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 9; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 9; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 28.054000; // C2H4
  CW += c[1] * 2.016000;  // H2
  CW += c[2] * 31.998000; // O2
  CW += c[3] * 18.015000; // H2O
  CW += c[4] * 28.010000; // CO
  CW += c[5] * 44.009000; // CO2
  CW += c[6] * 28.014000; // N2
  CW += c[7] * 4.002602;  // HE
  CW += c[8] * 39.950000; // AR
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 28.054000 * CWinv;
  y[1] = c[1] * 2.016000 * CWinv;
  y[2] = c[2] * 31.998000 * CWinv;
  y[3] = c[3] * 18.015000 * CWinv;
  y[4] = c[4] * 28.010000 * CWinv;
  y[5] = c[5] * 44.009000 * CWinv;
  y[6] = c[6] * 28.014000 * CWinv;
  y[7] = c[7] * 4.002602 * CWinv;
  y[8] = c[8] * 39.950000 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    cvml[id] *= 8.31446261815324e+07;
  }
}

// get specific heat at constant pressure as a
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPML(const amrex::Real T, amrex::Real cpml[])
{
  cp_R(cpml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    cpml[id] *= 8.31446261815324e+07;
  }
}

// get internal energy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUML(const amrex::Real T, amrex::Real uml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    uml[id] *= RT;
  }
}

// get enthalpy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHML(const amrex::Real T, amrex::Real hml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    sml[id] *= 8.31446261815324e+07;
  }
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
  cv_R(cvms, T);
  // multiply by R/molecularweight
  cvms[0] *= 2.963735160103101e+06; // C2H4
  cvms[1] *= 4.124237409798234e+07; // H2
  cvms[2] *= 2.598431970171023e+06; // O2
  cvms[3] *= 4.615299815794193e+06; // H2O
  cvms[4] *= 2.968390795484913e+06; // CO
  cvms[5] *= 1.889264154639560e+06; // CO2
  cvms[6] *= 2.967966951578939e+06; // N2
  cvms[7] *= 2.077264394049980e+07; // HE
  cvms[8] *= 2.081217176008320e+06; // AR
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 2.963735160103101e+06; // C2H4
  cpms[1] *= 4.124237409798234e+07; // H2
  cpms[2] *= 2.598431970171023e+06; // O2
  cpms[3] *= 4.615299815794193e+06; // H2O
  cpms[4] *= 2.968390795484913e+06; // CO
  cpms[5] *= 1.889264154639560e+06; // CO2
  cpms[6] *= 2.967966951578939e+06; // N2
  cpms[7] *= 2.077264394049980e+07; // HE
  cpms[8] *= 2.081217176008320e+06; // AR
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 9; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 9; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 2.963735160103101e+06; // C2H4
  sms[1] *= 4.124237409798234e+07; // H2
  sms[2] *= 2.598431970171023e+06; // O2
  sms[3] *= 4.615299815794193e+06; // H2O
  sms[4] *= 2.968390795484913e+06; // CO
  sms[5] *= 1.889264154639560e+06; // CO2
  sms[6] *= 2.967966951578939e+06; // N2
  sms[7] *= 2.077264394049980e+07; // HE
  sms[8] *= 2.081217176008320e+06; // AR
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* qf,
  amrex::Real* qr,
  const amrex::Real* sc,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real T,
  const amrex::Real invT,
  const amrex::Real logT)
{

  // reaction 0: C2H4 + O2 <=> 2 CO + 2 H2
  qf[0] = sc[0] * sc[2];
  qr[0] = (sc[1] * sc[1]) * (sc[4] * sc[4]);

  // reaction 1: 2 CO + O2 <=> 2 CO2
  qf[1] = sc[2] * (sc[4] * sc[4]);
  qr[1] = (sc[5] * sc[5]);

  // reaction 2: 2 H2 + O2 <=> 2 H2O
  qf[2] = (sc[1] * sc[1]) * sc[2];
  qr[2] = (sc[3] * sc[3]);

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 9; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;

  // reaction 0:  C2H4 + O2 <=> 2 CO + 2 H2
  k_f = 210000000 * exp(-(18015.2069574755) * invT);
  qf[0] *= k_f;
  qr[0] *= k_f *
           exp(-(g_RT[0] - 2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[4])) *
           ((refCinv * refCinv));
  // reaction 1:  2 CO + O2 <=> 2 CO2
  k_f = 0.348 * exp((2) * logT - (10134.8339709736) * invT);
  qf[1] *= k_f;
  qr[1] *=
    k_f * exp(-(g_RT[2] + 2.000000 * g_RT[4] - 2.000000 * g_RT[5])) * (refC);
  // reaction 2:  2 H2 + O2 <=> 2 H2O
  k_f = 300000000 * exp((-1) * logT);
  qf[2] *= k_f;
  qr[2] *=
    k_f * exp(-(2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[3])) * (refC);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 9; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 9; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  {
    // reaction 0:  C2H4 + O2 <=> 2 CO + 2 H2
    const amrex::Real k_f = 210000000 * exp(-(18015.2069574755) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[2]);
    const amrex::Real qr =
      k_f *
      exp(-(g_RT[0] - 2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[4])) *
      ((refCinv * refCinv)) * ((sc[1] * sc[1]) * (sc[4] * sc[4]));
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += 2.000000 * qdot;
    wdot[2] -= qdot;
    wdot[4] += 2.000000 * qdot;
  }

  {
    // reaction 1:  2 CO + O2 <=> 2 CO2
    const amrex::Real k_f = 0.348 * exp((2) * logT - (10134.8339709736) * invT);
    const amrex::Real qf = k_f * (sc[2] * (sc[4] * sc[4]));
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] + 2.000000 * g_RT[4] - 2.000000 * g_RT[5])) * (refC) *
      ((sc[5] * sc[5]));
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[4] -= 2.000000 * qdot;
    wdot[5] += 2.000000 * qdot;
  }

  {
    // reaction 2:  2 H2 + O2 <=> 2 H2O
    const amrex::Real k_f = 300000000 * exp((-1) * logT);
    const amrex::Real qf = k_f * ((sc[1] * sc[1]) * sc[2]);
    const amrex::Real qr =
      k_f * exp(-(2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[3])) * (refC) *
      ((sc[3] * sc[3]));
    const amrex::Real qdot = qf - qr;
    wdot[1] -= 2.000000 * qdot;
    wdot[2] -= qdot;
    wdot[3] += 2.000000 * qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 9; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    C[id] *= 1.0e-6;
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[9]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 9; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 9; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[9]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[9]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 9; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[9];   // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 28.054000; // C2H4
  XW += x[1] * 2.016000;  // H2
  XW += x[2] * 31.998000; // O2
  XW += x[3] * 18.015000; // H2O
  XW += x[4] * 28.010000; // CO
  XW += x[5] * 44.009000; // CO2
  XW += x[6] * 28.014000; // N2
  XW += x[7] * 4.002602;  // HE
  XW += x[8] * 39.950000; // AR
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 9; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 9; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // C2H4
  kcharge[1] = 0; // H2
  kcharge[2] = 0; // O2
  kcharge[3] = 0; // H2O
  kcharge[4] = 0; // CO
  kcharge[5] = 0; // CO2
  kcharge[6] = 0; // N2
  kcharge[7] = 0; // HE
  kcharge[8] = 0; // AR
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[9];
  CKCHRG(kchrg);

  for (int id = 0; id < 9; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 100; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[9];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 9; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[9];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[9];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: C2H4 + O2 <=> 2 CO + 2 H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 210000000 * exp(-(18015.2069574755) * invT);
  dlnkfdT = (18015.2069574755) * invT2;
  // reverse
  phi_r = (sc[1] * sc[1]) * (sc[4] * sc[4]);
  Kc = (refC * refC) *
       exp(g_RT[0] - 2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) +
                    (2.000000 * h_RT[1] + 2.000000 * h_RT[4]) - 2.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // C2H4
  wdot[1] += 2 * q; // H2
  wdot[2] -= q;     // O2
  wdot[4] += 2 * q; // CO
  // d()/d[C2H4]
  dqdci = +k_f * sc[2];
  J[0] -= dqdci;     // dwdot[C2H4]/d[C2H4]
  J[1] += 2 * dqdci; // dwdot[H2]/d[C2H4]
  J[2] -= dqdci;     // dwdot[O2]/d[C2H4]
  J[4] += 2 * dqdci; // dwdot[CO]/d[C2H4]
  // d()/d[H2]
  dqdci = -k_r * 2.000000 * sc[1] * sc[4] * sc[4];
  J[10] -= dqdci;     // dwdot[C2H4]/d[H2]
  J[11] += 2 * dqdci; // dwdot[H2]/d[H2]
  J[12] -= dqdci;     // dwdot[O2]/d[H2]
  J[14] += 2 * dqdci; // dwdot[CO]/d[H2]
  // d()/d[O2]
  dqdci = +k_f * sc[0];
  J[20] -= dqdci;     // dwdot[C2H4]/d[O2]
  J[21] += 2 * dqdci; // dwdot[H2]/d[O2]
  J[22] -= dqdci;     // dwdot[O2]/d[O2]
  J[24] += 2 * dqdci; // dwdot[CO]/d[O2]
  // d()/d[CO]
  dqdci = -k_r * sc[1] * sc[1] * 2.000000 * sc[4];
  J[40] -= dqdci;     // dwdot[C2H4]/d[CO]
  J[41] += 2 * dqdci; // dwdot[H2]/d[CO]
  J[42] -= dqdci;     // dwdot[O2]/d[CO]
  J[44] += 2 * dqdci; // dwdot[CO]/d[CO]
  // d()/dT
  J[90] -= dqdT;     // dwdot[C2H4]/dT
  J[91] += 2 * dqdT; // dwdot[H2]/dT
  J[92] -= dqdT;     // dwdot[O2]/dT
  J[94] += 2 * dqdT; // dwdot[CO]/dT

  // reaction 1: 2 CO + O2 <=> 2 CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * (sc[4] * sc[4]);
  k_f = 0.348 * exp(2 * logT - (10134.8339709736) * invT);
  dlnkfdT = 2 * invT + (10134.8339709736) * invT2;
  // reverse
  phi_r = (sc[5] * sc[5]);
  Kc = refCinv * exp(g_RT[2] + 2.000000 * g_RT[4] - 2.000000 * g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[2] + 2.000000 * h_RT[4]) + (2.000000 * h_RT[5]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;     // O2
  wdot[4] -= 2 * q; // CO
  wdot[5] += 2 * q; // CO2
  // d()/d[O2]
  dqdci = +k_f * sc[4] * sc[4];
  J[22] -= dqdci;      // dwdot[O2]/d[O2]
  J[24] += -2 * dqdci; // dwdot[CO]/d[O2]
  J[25] += 2 * dqdci;  // dwdot[CO2]/d[O2]
  // d()/d[CO]
  dqdci = +k_f * sc[2] * 2.000000 * sc[4];
  J[42] -= dqdci;      // dwdot[O2]/d[CO]
  J[44] += -2 * dqdci; // dwdot[CO]/d[CO]
  J[45] += 2 * dqdci;  // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * 2.000000 * sc[5];
  J[52] -= dqdci;      // dwdot[O2]/d[CO2]
  J[54] += -2 * dqdci; // dwdot[CO]/d[CO2]
  J[55] += 2 * dqdci;  // dwdot[CO2]/d[CO2]
  // d()/dT
  J[92] -= dqdT;      // dwdot[O2]/dT
  J[94] += -2 * dqdT; // dwdot[CO]/dT
  J[95] += 2 * dqdT;  // dwdot[CO2]/dT

  // reaction 2: 2 H2 + O2 <=> 2 H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[1] * sc[1]) * sc[2];
  k_f = 300000000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = (sc[3] * sc[3]);
  Kc = refCinv * exp(2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(2.000000 * h_RT[1] + h_RT[2]) + (2.000000 * h_RT[3]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= 2 * q; // H2
  wdot[2] -= q;     // O2
  wdot[3] += 2 * q; // H2O
  // d()/d[H2]
  dqdci = +k_f * 2.000000 * sc[1] * sc[2];
  J[11] += -2 * dqdci; // dwdot[H2]/d[H2]
  J[12] -= dqdci;      // dwdot[O2]/d[H2]
  J[13] += 2 * dqdci;  // dwdot[H2O]/d[H2]
  // d()/d[O2]
  dqdci = +k_f * sc[1] * sc[1];
  J[21] += -2 * dqdci; // dwdot[H2]/d[O2]
  J[22] -= dqdci;      // dwdot[O2]/d[O2]
  J[23] += 2 * dqdci;  // dwdot[H2O]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * 2.000000 * sc[3];
  J[31] += -2 * dqdci; // dwdot[H2]/d[H2O]
  J[32] -= dqdci;      // dwdot[O2]/d[H2O]
  J[33] += 2 * dqdci;  // dwdot[H2O]/d[H2O]
  // d()/dT
  J[91] += -2 * dqdT; // dwdot[H2]/dT
  J[92] -= dqdT;      // dwdot[O2]/dT
  J[93] += 2 * dqdT;  // dwdot[H2O]/dT

  amrex::Real c_R[9], dcRdT[9], e_RT[9];
  amrex::Real* eh_RT;
  if (HP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 9; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[90 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 9; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 9; ++m) {
      dehmixdc += eh_RT[m] * J[k * 10 + m];
    }
    J[k * 10 + 9] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[99] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[9];

  for (int k = 0; k < 9; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 9; k++) {
    J[90 + k] *= 1.e-6;
    J[k * 10 + 9] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 100; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[9];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 9; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[9];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[9];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[9];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: C2H4 + O2 <=> 2 CO + 2 H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 210000000 * exp(-(18015.2069574755) * invT);
  dlnkfdT = (18015.2069574755) * invT2;
  // reverse
  phi_r = (sc[1] * sc[1]) * (sc[4] * sc[4]);
  Kc = (refC * refC) *
       exp(g_RT[0] - 2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) +
                    (2.000000 * h_RT[1] + 2.000000 * h_RT[4]) - 2.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // C2H4
  wdot[1] += 2 * q; // H2
  wdot[2] -= q;     // O2
  wdot[4] += 2 * q; // CO
  // d()/d[C2H4]
  dqdci = +k_f * sc[2];
  J[0] -= dqdci;     // dwdot[C2H4]/d[C2H4]
  J[1] += 2 * dqdci; // dwdot[H2]/d[C2H4]
  J[2] -= dqdci;     // dwdot[O2]/d[C2H4]
  J[4] += 2 * dqdci; // dwdot[CO]/d[C2H4]
  // d()/d[H2]
  dqdci = -k_r * 2.000000 * sc[1] * sc[4] * sc[4];
  J[10] -= dqdci;     // dwdot[C2H4]/d[H2]
  J[11] += 2 * dqdci; // dwdot[H2]/d[H2]
  J[12] -= dqdci;     // dwdot[O2]/d[H2]
  J[14] += 2 * dqdci; // dwdot[CO]/d[H2]
  // d()/d[O2]
  dqdci = +k_f * sc[0];
  J[20] -= dqdci;     // dwdot[C2H4]/d[O2]
  J[21] += 2 * dqdci; // dwdot[H2]/d[O2]
  J[22] -= dqdci;     // dwdot[O2]/d[O2]
  J[24] += 2 * dqdci; // dwdot[CO]/d[O2]
  // d()/d[CO]
  dqdci = -k_r * sc[1] * sc[1] * 2.000000 * sc[4];
  J[40] -= dqdci;     // dwdot[C2H4]/d[CO]
  J[41] += 2 * dqdci; // dwdot[H2]/d[CO]
  J[42] -= dqdci;     // dwdot[O2]/d[CO]
  J[44] += 2 * dqdci; // dwdot[CO]/d[CO]
  // d()/dT
  J[90] -= dqdT;     // dwdot[C2H4]/dT
  J[91] += 2 * dqdT; // dwdot[H2]/dT
  J[92] -= dqdT;     // dwdot[O2]/dT
  J[94] += 2 * dqdT; // dwdot[CO]/dT

  // reaction 1: 2 CO + O2 <=> 2 CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * (sc[4] * sc[4]);
  k_f = 0.348 * exp(2 * logT - (10134.8339709736) * invT);
  dlnkfdT = 2 * invT + (10134.8339709736) * invT2;
  // reverse
  phi_r = (sc[5] * sc[5]);
  Kc = refCinv * exp(g_RT[2] + 2.000000 * g_RT[4] - 2.000000 * g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[2] + 2.000000 * h_RT[4]) + (2.000000 * h_RT[5]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;     // O2
  wdot[4] -= 2 * q; // CO
  wdot[5] += 2 * q; // CO2
  // d()/d[O2]
  dqdci = +k_f * sc[4] * sc[4];
  J[22] -= dqdci;      // dwdot[O2]/d[O2]
  J[24] += -2 * dqdci; // dwdot[CO]/d[O2]
  J[25] += 2 * dqdci;  // dwdot[CO2]/d[O2]
  // d()/d[CO]
  dqdci = +k_f * sc[2] * 2.000000 * sc[4];
  J[42] -= dqdci;      // dwdot[O2]/d[CO]
  J[44] += -2 * dqdci; // dwdot[CO]/d[CO]
  J[45] += 2 * dqdci;  // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * 2.000000 * sc[5];
  J[52] -= dqdci;      // dwdot[O2]/d[CO2]
  J[54] += -2 * dqdci; // dwdot[CO]/d[CO2]
  J[55] += 2 * dqdci;  // dwdot[CO2]/d[CO2]
  // d()/dT
  J[92] -= dqdT;      // dwdot[O2]/dT
  J[94] += -2 * dqdT; // dwdot[CO]/dT
  J[95] += 2 * dqdT;  // dwdot[CO2]/dT

  // reaction 2: 2 H2 + O2 <=> 2 H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[1] * sc[1]) * sc[2];
  k_f = 300000000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = (sc[3] * sc[3]);
  Kc = refCinv * exp(2.000000 * g_RT[1] + g_RT[2] - 2.000000 * g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(2.000000 * h_RT[1] + h_RT[2]) + (2.000000 * h_RT[3]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= 2 * q; // H2
  wdot[2] -= q;     // O2
  wdot[3] += 2 * q; // H2O
  // d()/d[H2]
  dqdci = +k_f * 2.000000 * sc[1] * sc[2];
  J[11] += -2 * dqdci; // dwdot[H2]/d[H2]
  J[12] -= dqdci;      // dwdot[O2]/d[H2]
  J[13] += 2 * dqdci;  // dwdot[H2O]/d[H2]
  // d()/d[O2]
  dqdci = +k_f * sc[1] * sc[1];
  J[21] += -2 * dqdci; // dwdot[H2]/d[O2]
  J[22] -= dqdci;      // dwdot[O2]/d[O2]
  J[23] += 2 * dqdci;  // dwdot[H2O]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * 2.000000 * sc[3];
  J[31] += -2 * dqdci; // dwdot[H2]/d[H2O]
  J[32] -= dqdci;      // dwdot[O2]/d[H2O]
  J[33] += 2 * dqdci;  // dwdot[H2O]/d[H2O]
  // d()/dT
  J[91] += -2 * dqdT; // dwdot[H2]/dT
  J[92] -= dqdT;      // dwdot[O2]/dT
  J[93] += 2 * dqdT;  // dwdot[H2O]/dT

  amrex::Real c_R[9], dcRdT[9], e_RT[9];
  amrex::Real* eh_RT;
  if (consP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 9; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[90 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 9; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 9; ++m) {
      dehmixdc += eh_RT[m] * J[k * 10 + m];
    }
    J[k * 10 + 9] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[99] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT(
  amrex::Real* J,
  const amrex::Real* sc,
  const amrex::Real* Tp,
  const int* consP)
{
  amrex::Real c[9];

  for (int k = 0; k < 9; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 9; k++) {
    J[90 + k] *= 1.e-6;
    J[k * 10 + 9] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 38;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 1854;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 9;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 2;
}

// Patm in ergs/cm3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPATM(amrex::Real* PATM)
{
  *PATM = 0.1013250000000000E+07;
}

// the molecular weights in g/mol
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetWT(amrex::Real* WT)
{
  WT[0] = 2.80540000E+01;
  WT[1] = 2.01600000E+00;
  WT[2] = 3.19980000E+01;
  WT[3] = 1.80150000E+01;
  WT[4] = 2.80100000E+01;
  WT[5] = 4.40090000E+01;
  WT[6] = 2.80140000E+01;
  WT[7] = 4.00260200E+00;
  WT[8] = 3.99500000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 2.80800000E+02;
  EPS[1] = 3.80000000E+01;
  EPS[2] = 1.07400000E+02;
  EPS[3] = 5.72400000E+02;
  EPS[4] = 9.81000000E+01;
  EPS[5] = 2.44000000E+02;
  EPS[6] = 9.75300000E+01;
  EPS[7] = 1.02000000E+01;
  EPS[8] = 1.36500000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.97100000E+00;
  SIG[1] = 2.92000000E+00;
  SIG[2] = 3.45800000E+00;
  SIG[3] = 2.60500000E+00;
  SIG[4] = 3.65000000E+00;
  SIG[5] = 3.76300000E+00;
  SIG[6] = 3.62100000E+00;
  SIG[7] = 2.57600000E+00;
  SIG[8] = 3.33000000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 1.84400000E+00;
  DIP[4] = 1.10000000E-01;
  DIP[5] = 0.00000000E+00;
  DIP[6] = 0.00000000E+00;
  DIP[7] = 0.00000000E+00;
  DIP[8] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 0.00000000E+00;
  POL[1] = 7.90000000E-01;
  POL[2] = 1.60000000E+00;
  POL[3] = 0.00000000E+00;
  POL[4] = 1.95000000E+00;
  POL[5] = 2.65000000E+00;
  POL[6] = 1.76000000E+00;
  POL[7] = 0.00000000E+00;
  POL[8] = 0.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 1.50000000E+00;
  ZROT[1] = 2.80000000E+02;
  ZROT[2] = 3.80000000E+00;
  ZROT[3] = 4.00000000E+00;
  ZROT[4] = 1.80000000E+00;
  ZROT[5] = 2.10000000E+00;
  ZROT[6] = 4.00000000E+00;
  ZROT[7] = 0.00000000E+00;
  ZROT[8] = 0.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 2;
  NLIN[1] = 1;
  NLIN[2] = 1;
  NLIN[3] = 2;
  NLIN[4] = 1;
  NLIN[5] = 1;
  NLIN[6] = 1;
  NLIN[7] = 0;
  NLIN[8] = 0;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -2.29541132E+01;
  COFETA[1] = 4.43807276E+00;
  COFETA[2] = -4.62324516E-01;
  COFETA[3] = 1.87855972E-02;
  COFETA[4] = -1.40419383E+01;
  COFETA[5] = 1.08789225E+00;
  COFETA[6] = -6.18592115E-02;
  COFETA[7] = 2.86838304E-03;
  COFETA[8] = -1.78915955E+01;
  COFETA[9] = 2.98311502E+00;
  COFETA[10] = -3.14105508E-01;
  COFETA[11] = 1.40500162E-02;
  COFETA[12] = -1.14441704E+01;
  COFETA[13] = -9.67162726E-01;
  COFETA[14] = 3.58651315E-01;
  COFETA[15] = -2.09789188E-02;
  COFETA[16] = -1.74469941E+01;
  COFETA[17] = 2.74723873E+00;
  COFETA[18] = -2.83498165E-01;
  COFETA[19] = 1.27260303E-02;
  COFETA[20] = -2.28110458E+01;
  COFETA[21] = 4.62954710E+00;
  COFETA[22] = -5.00689001E-01;
  COFETA[23] = 2.10012969E-02;
  COFETA[24] = -1.73976840E+01;
  COFETA[25] = 2.73482764E+00;
  COFETA[26] = -2.81916034E-01;
  COFETA[27] = 1.26588405E-02;
  COFETA[28] = -1.19627938E+01;
  COFETA[29] = 5.61503354E-01;
  COFETA[30] = 7.80356928E-03;
  COFETA[31] = -1.10734424E-04;
  COFETA[32] = -1.95595455E+01;
  COFETA[33] = 3.68007692E+00;
  COFETA[34] = -4.03161815E-01;
  COFETA[35] = 1.78455135E-02;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -4.03620887E+00;
  COFLAM[1] = 1.90761262E+00;
  COFLAM[2] = 1.17538704E-01;
  COFLAM[3] = -1.61049430E-02;
  COFLAM[4] = 4.17091754E+00;
  COFLAM[5] = 1.62804826E+00;
  COFLAM[6] = -1.70908200E-01;
  COFLAM[7] = 1.03632992E-02;
  COFLAM[8] = 7.13715609E-01;
  COFLAM[9] = 1.79167550E+00;
  COFLAM[10] = -1.20340496E-01;
  COFLAM[11] = 4.69808739E-03;
  COFLAM[12] = 1.78984926E+01;
  COFLAM[13] = -6.62285295E+00;
  COFLAM[14] = 1.19609125E+00;
  COFLAM[15] = -6.02372587E-02;
  COFLAM[16] = 8.25303418E+00;
  COFLAM[17] = -1.56824868E+00;
  COFLAM[18] = 3.71642581E-01;
  COFLAM[19] = -1.92255419E-02;
  COFLAM[20] = -8.59234014E+00;
  COFLAM[21] = 4.72899549E+00;
  COFLAM[22] = -4.10258362E-01;
  COFLAM[23] = 1.31618557E-02;
  COFLAM[24] = 7.77703418E+00;
  COFLAM[25] = -1.30958100E+00;
  COFLAM[26] = 3.28842904E-01;
  COFLAM[27] = -1.69485811E-02;
  COFLAM[28] = 6.20811043E+00;
  COFLAM[29] = 5.61503354E-01;
  COFLAM[30] = 7.80356928E-03;
  COFLAM[31] = -1.10734424E-04;
  COFLAM[32] = -3.68932526E+00;
  COFLAM[33] = 3.68007692E+00;
  COFLAM[34] = -4.03161815E-01;
  COFLAM[35] = 1.78455135E-02;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -2.04987774E+01;
  COFD[1] = 4.99303150E+00;
  COFD[2] = -4.04298713E-01;
  COFD[3] = 1.63496647E-02;
  COFD[4] = -1.47832963E+01;
  COFD[5] = 3.61918603E+00;
  COFD[6] = -2.60701094E-01;
  COFD[7] = 1.15292870E-02;
  COFD[8] = -1.91360384E+01;
  COFD[9] = 4.75557446E+00;
  COFD[10] = -4.00539622E-01;
  COFD[11] = 1.72785461E-02;
  COFD[12] = -1.83508955E+01;
  COFD[13] = 4.02222416E+00;
  COFD[14] = -2.41671549E-01;
  COFD[15] = 7.85284958E-03;
  COFD[16] = -1.88433114E+01;
  COFD[17] = 4.64610663E+00;
  COFD[18] = -3.87397753E-01;
  COFD[19] = 1.67516180E-02;
  COFD[20] = -2.05833306E+01;
  COFD[21] = 5.05774582E+00;
  COFD[22] = -4.18194338E-01;
  COFD[23] = 1.71715531E-02;
  COFD[24] = -1.88223317E+01;
  COFD[25] = 4.64181074E+00;
  COFD[26] = -3.86929312E-01;
  COFD[27] = 1.67349243E-02;
  COFD[28] = -1.17734917E+01;
  COFD[29] = 2.38607269E+00;
  COFD[30] = -9.77163599E-02;
  COFD[31] = 4.33434581E-03;
  COFD[32] = -1.96932021E+01;
  COFD[33] = 4.92080280E+00;
  COFD[34] = -4.17253648E-01;
  COFD[35] = 1.78096976E-02;
  COFD[36] = -1.47832963E+01;
  COFD[37] = 3.61918603E+00;
  COFD[38] = -2.60701094E-01;
  COFD[39] = 1.15292870E-02;
  COFD[40] = -1.04285243E+01;
  COFD[41] = 2.23477534E+00;
  COFD[42] = -8.11809423E-02;
  COFD[43] = 3.77342041E-03;
  COFD[44] = -1.22181330E+01;
  COFD[45] = 2.70415313E+00;
  COFD[46] = -1.41236971E-01;
  COFD[47] = 6.32236816E-03;
  COFD[48] = -1.73864192E+01;
  COFD[49] = 4.71143120E+00;
  COFD[50] = -3.95288666E-01;
  COFD[51] = 1.70702288E-02;
  COFD[52] = -1.20630806E+01;
  COFD[53] = 2.62057553E+00;
  COFD[54] = -1.29759407E-01;
  COFD[55] = 5.79604780E-03;
  COFD[56] = -1.43978813E+01;
  COFD[57] = 3.49721576E+00;
  COFD[58] = -2.45465191E-01;
  COFD[59] = 1.08948372E-02;
  COFD[60] = -1.20381551E+01;
  COFD[61] = 2.61421687E+00;
  COFD[62] = -1.28887086E-01;
  COFD[63] = 5.75609167E-03;
  COFD[64] = -9.65850921E+00;
  COFD[65] = 1.96415709E+00;
  COFD[66] = -4.61770517E-02;
  COFD[67] = 2.28331169E-03;
  COFD[68] = -1.27750586E+01;
  COFD[69] = 2.93486520E+00;
  COFD[70] = -1.72251939E-01;
  COFD[71] = 7.71404703E-03;
  COFD[72] = -1.91360384E+01;
  COFD[73] = 4.75557446E+00;
  COFD[74] = -4.00539622E-01;
  COFD[75] = 1.72785461E-02;
  COFD[76] = -1.22181330E+01;
  COFD[77] = 2.70415313E+00;
  COFD[78] = -1.41236971E-01;
  COFD[79] = 6.32236816E-03;
  COFD[80] = -1.60936459E+01;
  COFD[81] = 3.70633871E+00;
  COFD[82] = -2.71897253E-01;
  COFD[83] = 1.20097588E-02;
  COFD[84] = -1.99035490E+01;
  COFD[85] = 5.01694632E+00;
  COFD[86] = -4.08962988E-01;
  COFD[87] = 1.66143402E-02;
  COFD[88] = -1.58477684E+01;
  COFD[89] = 3.60670618E+00;
  COFD[90] = -2.59109524E-01;
  COFD[91] = 1.14615052E-02;
  COFD[92] = -1.87633989E+01;
  COFD[93] = 4.61060397E+00;
  COFD[94] = -3.83564503E-01;
  COFD[95] = 1.66168246E-02;
  COFD[96] = -1.58214949E+01;
  COFD[97] = 3.60000113E+00;
  COFD[98] = -2.58255120E-01;
  COFD[99] = 1.14251480E-02;
  COFD[100] = -1.06317398E+01;
  COFD[101] = 2.03312891E+00;
  COFD[102] = -5.25105447E-02;
  COFD[103] = 2.41904369E-03;
  COFD[104] = -1.68526550E+01;
  COFD[105] = 3.98037000E+00;
  COFD[106] = -3.06459320E-01;
  COFD[107] = 1.34649035E-02;
  COFD[108] = -1.83508955E+01;
  COFD[109] = 4.02222416E+00;
  COFD[110] = -2.41671549E-01;
  COFD[111] = 7.85284958E-03;
  COFD[112] = -1.73864192E+01;
  COFD[113] = 4.71143120E+00;
  COFD[114] = -3.95288666E-01;
  COFD[115] = 1.70702288E-02;
  COFD[116] = -1.99035490E+01;
  COFD[117] = 5.01694632E+00;
  COFD[118] = -4.08962988E-01;
  COFD[119] = 1.66143402E-02;
  COFD[120] = -1.16123739E+01;
  COFD[121] = 8.27753087E-01;
  COFD[122] = 2.52262500E-01;
  COFD[123] = -1.62567542E-02;
  COFD[124] = -1.97211208E+01;
  COFD[125] = 5.03205924E+00;
  COFD[126] = -4.21097978E-01;
  COFD[127] = 1.75482764E-02;
  COFD[128] = -1.82187534E+01;
  COFD[129] = 3.93854125E+00;
  COFD[130] = -2.28424577E-01;
  COFD[131] = 7.18603064E-03;
  COFD[132] = -1.99472346E+01;
  COFD[133] = 5.05636620E+00;
  COFD[134] = -4.17733674E-01;
  COFD[135] = 1.71403498E-02;
  COFD[136] = -1.29473745E+01;
  COFD[137] = 3.04220308E+00;
  COFD[138] = -1.86434897E-01;
  COFD[139] = 8.33937065E-03;
  COFD[140] = -1.98625079E+01;
  COFD[141] = 4.99946719E+00;
  COFD[142] = -4.05534057E-01;
  COFD[143] = 1.64193031E-02;
  COFD[144] = -1.88433114E+01;
  COFD[145] = 4.64610663E+00;
  COFD[146] = -3.87397753E-01;
  COFD[147] = 1.67516180E-02;
  COFD[148] = -1.20630806E+01;
  COFD[149] = 2.62057553E+00;
  COFD[150] = -1.29759407E-01;
  COFD[151] = 5.79604780E-03;
  COFD[152] = -1.58477684E+01;
  COFD[153] = 3.60670618E+00;
  COFD[154] = -2.59109524E-01;
  COFD[155] = 1.14615052E-02;
  COFD[156] = -1.97211208E+01;
  COFD[157] = 5.03205924E+00;
  COFD[158] = -4.21097978E-01;
  COFD[159] = 1.75482764E-02;
  COFD[160] = -1.56427954E+01;
  COFD[161] = 3.52431900E+00;
  COFD[162] = -2.48773265E-01;
  COFD[163] = 1.10291160E-02;
  COFD[164] = -1.85342543E+01;
  COFD[165] = 4.52811654E+00;
  COFD[166] = -3.73924082E-01;
  COFD[167] = 1.62415273E-02;
  COFD[168] = -1.56236593E+01;
  COFD[169] = 3.52028384E+00;
  COFD[170] = -2.48273649E-01;
  COFD[171] = 1.10084878E-02;
  COFD[172] = -1.05788599E+01;
  COFD[173] = 1.99005948E+00;
  COFD[174] = -4.63757226E-02;
  COFD[175] = 2.12803780E-03;
  COFD[176] = -1.66160564E+01;
  COFD[177] = 3.88881928E+00;
  COFD[178] = -2.95160663E-01;
  COFD[179] = 1.30000777E-02;
  COFD[180] = -2.05833306E+01;
  COFD[181] = 5.05774582E+00;
  COFD[182] = -4.18194338E-01;
  COFD[183] = 1.71715531E-02;
  COFD[184] = -1.43978813E+01;
  COFD[185] = 3.49721576E+00;
  COFD[186] = -2.45465191E-01;
  COFD[187] = 1.08948372E-02;
  COFD[188] = -1.87633989E+01;
  COFD[189] = 4.61060397E+00;
  COFD[190] = -3.83564503E-01;
  COFD[191] = 1.66168246E-02;
  COFD[192] = -1.82187534E+01;
  COFD[193] = 3.93854125E+00;
  COFD[194] = -2.28424577E-01;
  COFD[195] = 7.18603064E-03;
  COFD[196] = -1.85342543E+01;
  COFD[197] = 4.52811654E+00;
  COFD[198] = -3.73924082E-01;
  COFD[199] = 1.62415273E-02;
  COFD[200] = -2.05810575E+01;
  COFD[201] = 5.07469434E+00;
  COFD[202] = -4.25340301E-01;
  COFD[203] = 1.76800795E-02;
  COFD[204] = -1.85068911E+01;
  COFD[205] = 4.52122572E+00;
  COFD[206] = -3.73088946E-01;
  COFD[207] = 1.62076520E-02;
  COFD[208] = -1.15280100E+01;
  COFD[209] = 2.30939755E+00;
  COFD[210] = -8.75686489E-02;
  COFD[211] = 3.88623174E-03;
  COFD[212] = -1.94864550E+01;
  COFD[213] = 4.84615884E+00;
  COFD[214] = -4.10706292E-01;
  COFD[215] = 1.76546483E-02;
  COFD[216] = -1.88223317E+01;
  COFD[217] = 4.64181074E+00;
  COFD[218] = -3.86929312E-01;
  COFD[219] = 1.67349243E-02;
  COFD[220] = -1.20381551E+01;
  COFD[221] = 2.61421687E+00;
  COFD[222] = -1.28887086E-01;
  COFD[223] = 5.75609167E-03;
  COFD[224] = -1.58214949E+01;
  COFD[225] = 3.60000113E+00;
  COFD[226] = -2.58255120E-01;
  COFD[227] = 1.14251480E-02;
  COFD[228] = -1.99472346E+01;
  COFD[229] = 5.05636620E+00;
  COFD[230] = -4.17733674E-01;
  COFD[231] = 1.71403498E-02;
  COFD[232] = -1.56236593E+01;
  COFD[233] = 3.52028384E+00;
  COFD[234] = -2.48273649E-01;
  COFD[235] = 1.10084878E-02;
  COFD[236] = -1.85068911E+01;
  COFD[237] = 4.52122572E+00;
  COFD[238] = -3.73088946E-01;
  COFD[239] = 1.62076520E-02;
  COFD[240] = -1.56019684E+01;
  COFD[241] = 3.51542686E+00;
  COFD[242] = -2.47677471E-01;
  COFD[243] = 1.09841319E-02;
  COFD[244] = -1.05656010E+01;
  COFD[245] = 1.98864138E+00;
  COFD[246] = -4.61820489E-02;
  COFD[247] = 2.11914887E-03;
  COFD[248] = -1.65903404E+01;
  COFD[249] = 3.88246200E+00;
  COFD[250] = -2.94366335E-01;
  COFD[251] = 1.29669592E-02;
  COFD[252] = -1.17734917E+01;
  COFD[253] = 2.38607269E+00;
  COFD[254] = -9.77163599E-02;
  COFD[255] = 4.33434581E-03;
  COFD[256] = -9.65850921E+00;
  COFD[257] = 1.96415709E+00;
  COFD[258] = -4.61770517E-02;
  COFD[259] = 2.28331169E-03;
  COFD[260] = -1.06317398E+01;
  COFD[261] = 2.03312891E+00;
  COFD[262] = -5.25105447E-02;
  COFD[263] = 2.41904369E-03;
  COFD[264] = -1.29473745E+01;
  COFD[265] = 3.04220308E+00;
  COFD[266] = -1.86434897E-01;
  COFD[267] = 8.33937065E-03;
  COFD[268] = -1.05788599E+01;
  COFD[269] = 1.99005948E+00;
  COFD[270] = -4.63757226E-02;
  COFD[271] = 2.12803780E-03;
  COFD[272] = -1.15280100E+01;
  COFD[273] = 2.30939755E+00;
  COFD[274] = -8.75686489E-02;
  COFD[275] = 3.88623174E-03;
  COFD[276] = -1.05656010E+01;
  COFD[277] = 1.98864138E+00;
  COFD[278] = -4.61820489E-02;
  COFD[279] = 2.11914887E-03;
  COFD[280] = -8.46721849E+00;
  COFD[281] = 1.45185564E+00;
  COFD[282] = 2.82248624E-02;
  COFD[283] = -1.26887629E-03;
  COFD[284] = -1.10359245E+01;
  COFD[285] = 2.21197910E+00;
  COFD[286] = -7.79849799E-02;
  COFD[287] = 3.62466588E-03;
  COFD[288] = -1.96932021E+01;
  COFD[289] = 4.92080280E+00;
  COFD[290] = -4.17253648E-01;
  COFD[291] = 1.78096976E-02;
  COFD[292] = -1.27750586E+01;
  COFD[293] = 2.93486520E+00;
  COFD[294] = -1.72251939E-01;
  COFD[295] = 7.71404703E-03;
  COFD[296] = -1.68526550E+01;
  COFD[297] = 3.98037000E+00;
  COFD[298] = -3.06459320E-01;
  COFD[299] = 1.34649035E-02;
  COFD[300] = -1.98625079E+01;
  COFD[301] = 4.99946719E+00;
  COFD[302] = -4.05534057E-01;
  COFD[303] = 1.64193031E-02;
  COFD[304] = -1.66160564E+01;
  COFD[305] = 3.88881928E+00;
  COFD[306] = -2.95160663E-01;
  COFD[307] = 1.30000777E-02;
  COFD[308] = -1.94864550E+01;
  COFD[309] = 4.84615884E+00;
  COFD[310] = -4.10706292E-01;
  COFD[311] = 1.76546483E-02;
  COFD[312] = -1.65903404E+01;
  COFD[313] = 3.88246200E+00;
  COFD[314] = -2.94366335E-01;
  COFD[315] = 1.29669592E-02;
  COFD[316] = -1.10359245E+01;
  COFD[317] = 2.21197910E+00;
  COFD[318] = -7.79849799E-02;
  COFD[319] = 3.62466588E-03;
  COFD[320] = -1.76146738E+01;
  COFD[321] = 4.25063574E+00;
  COFD[322] = -3.40242318E-01;
  COFD[323] = 1.48768069E-02;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* KTDIF)
{
  KTDIF[0] = 1;
  KTDIF[1] = 7;
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* COFTD)
{
  COFTD[0] = 2.06620011E-01;
  COFTD[1] = 4.69836642E-04;
  COFTD[2] = -2.39995296E-07;
  COFTD[3] = 3.71484694E-11;
  COFTD[4] = 0.00000000E+00;
  COFTD[5] = 0.00000000E+00;
  COFTD[6] = 0.00000000E+00;
  COFTD[7] = 0.00000000E+00;
  COFTD[8] = 3.81861526E-01;
  COFTD[9] = 1.84116078E-04;
  COFTD[10] = -9.79610689E-08;
  COFTD[11] = 1.62541101E-11;
  COFTD[12] = 1.95116681E-03;
  COFTD[13] = 6.69463662E-04;
  COFTD[14] = -3.12145327E-07;
  COFTD[15] = 4.52944954E-11;
  COFTD[16] = 3.87278497E-01;
  COFTD[17] = 1.57125093E-04;
  COFTD[18] = -8.30661692E-08;
  COFTD[19] = 1.38677574E-11;
  COFTD[20] = 2.47127846E-01;
  COFTD[21] = 4.49393559E-04;
  COFTD[22] = -2.32029646E-07;
  COFTD[23] = 3.62577088E-11;
  COFTD[24] = 3.88180656E-01;
  COFTD[25] = 1.55380030E-04;
  COFTD[26] = -8.20879923E-08;
  COFTD[27] = 1.37104470E-11;
  COFTD[28] = 1.68772245E-01;
  COFTD[29] = 3.41606317E-05;
  COFTD[30] = -9.74466932E-09;
  COFTD[31] = -2.98780942E-12;
  COFTD[32] = 3.53882993E-01;
  COFTD[33] = 2.59421279E-04;
  COFTD[34] = -1.38280863E-07;
  COFTD[35] = 2.25371639E-11;
  COFTD[36] = 3.63175857E-01;
  COFTD[37] = 8.10628901E-05;
  COFTD[38] = -4.01873782E-08;
  COFTD[39] = 6.77340318E-12;
  COFTD[40] = -1.68772245E-01;
  COFTD[41] = -3.41606317E-05;
  COFTD[42] = 9.74466932E-09;
  COFTD[43] = 2.98780942E-12;
  COFTD[44] = 4.27359094E-01;
  COFTD[45] = -2.44276852E-05;
  COFTD[46] = 3.39664655E-08;
  COFTD[47] = -8.27712844E-12;
  COFTD[48] = 2.34820729E-01;
  COFTD[49] = 2.08493877E-04;
  COFTD[50] = -1.10710615E-07;
  COFTD[51] = 1.78835096E-11;
  COFTD[52] = 4.12652469E-01;
  COFTD[53] = -2.40321785E-05;
  COFTD[54] = 3.45963874E-08;
  COFTD[55] = -8.78322135E-12;
  COFTD[56] = 4.16938456E-01;
  COFTD[57] = 6.18293981E-05;
  COFTD[58] = -2.74969374E-08;
  COFTD[59] = 4.56258305E-12;
  COFTD[60] = 4.12680021E-01;
  COFTD[61] = -2.40030605E-05;
  COFTD[62] = 3.46790648E-08;
  COFTD[63] = -8.83042357E-12;
  COFTD[64] = 0.00000000E+00;
  COFTD[65] = 0.00000000E+00;
  COFTD[66] = 0.00000000E+00;
  COFTD[67] = 0.00000000E+00;
  COFTD[68] = 4.44353431E-01;
  COFTD[69] = -1.51812726E-05;
  COFTD[70] = 2.50632330E-08;
  COFTD[71] = -5.73309337E-12;
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[9];
  amrex::Real SIG[9];
  amrex::Real wt[9];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !
  amrex::Real avogadro = 6.02214199e23;
  amrex::Real boltzmann = 1.3806503e-16; // we work in CGS

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: C2H4
  // Imported from NIST
  Tci[0] = 282.340000;
  ai[0] = 1e6 * 0.42748 * Rcst * Rcst * Tci[0] * Tci[0] /
          (28.054000 * 28.054000 * 50.410000);
  bi[0] = 0.08664 * Rcst * Tci[0] / (28.054000 * 50.410000);
  acentric_i[0] = 0.087000;

  // species 1: H2
  // Imported from NIST
  Tci[1] = 33.145000;
  ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] /
          (2.015880 * 2.015880 * 12.964000);
  bi[1] = 0.08664 * Rcst * Tci[1] / (2.015880 * 12.964000);
  acentric_i[1] = -0.219000;

  // species 2: O2
  // Imported from NIST
  Tci[2] = 154.581000;
  ai[2] = 1e6 * 0.42748 * Rcst * Rcst * Tci[2] * Tci[2] /
          (31.998800 * 31.998800 * 50.430466);
  bi[2] = 0.08664 * Rcst * Tci[2] / (31.998800 * 50.430466);
  acentric_i[2] = 0.022200;

  // species 3: H2O
  // Imported from NIST
  Tci[3] = 647.096000;
  ai[3] = 1e6 * 0.42748 * Rcst * Rcst * Tci[3] * Tci[3] /
          (18.015340 * 18.015340 * 220.640000);
  bi[3] = 0.08664 * Rcst * Tci[3] / (18.015340 * 220.640000);
  acentric_i[3] = 0.344300;

  // species 4: CO
  // Imported from NIST
  Tci[4] = 132.850000;
  ai[4] = 1e6 * 0.42748 * Rcst * Rcst * Tci[4] * Tci[4] /
          (28.010000 * 28.010000 * 34.940000);
  bi[4] = 0.08664 * Rcst * Tci[4] / (28.010000 * 34.940000);
  acentric_i[4] = 0.045000;

  // species 5: CO2
  // Imported from NIST
  Tci[5] = 304.120000;
  ai[5] = 1e6 * 0.42748 * Rcst * Rcst * Tci[5] * Tci[5] /
          (44.009950 * 44.009950 * 73.740000);
  bi[5] = 0.08664 * Rcst * Tci[5] / (44.009950 * 73.740000);
  acentric_i[5] = 0.225000;

  // species 6: N2
  // Imported from NIST
  Tci[6] = 126.192000;
  ai[6] = 1e6 * 0.42748 * Rcst * Rcst * Tci[6] * Tci[6] /
          (28.013400 * 28.013400 * 33.958000);
  bi[6] = 0.08664 * Rcst * Tci[6] / (28.013400 * 33.958000);
  acentric_i[6] = 0.037200;

  // species 7: HE
  Tci[7] = 1.316 * EPS[7];
  ai[7] = (5.55 * avogadro * avogadro * EPS[7] * boltzmann * 1e-24 * SIG[7] *
           SIG[7] * SIG[7]) /
          (wt[7] * wt[7]);
  bi[7] = 0.855 * avogadro * 1e-24 * SIG[7] * SIG[7] * SIG[7] / (wt[7]);
  acentric_i[7] = 0.0;

  // species 8: AR
  // Imported from NIST
  Tci[8] = 150.860000;
  ai[8] = 1e6 * 0.42748 * Rcst * Rcst * Tci[8] * Tci[8] /
          (39.948000 * 39.948000 * 48.980000);
  bi[8] = 0.08664 * Rcst * Tci[8] / (39.948000 * 48.980000);
  acentric_i[8] = -0.002000;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: C2H4
  // Imported from NIST
  sqrtOneOverTc[0] = 5.9513267432530e-02;
  sqrtAsti[0] = 7.7056842134487e+04;
  Bi[0] = 1.4381805192666e+00;
  Fomega[0] = 6.1893034120300e-01;

  // species 1: H2
  // Imported from NIST
  sqrtOneOverTc[1] = 1.7369646834812e-01;
  sqrtAsti[1] = 2.4824229305911e+05;
  Bi[1] = 9.1362254169923e+00;
  Fomega[1] = 1.3798618890700e-01;

  // species 2: O2
  // Imported from NIST
  sqrtOneOverTc[2] = 8.0430717653241e-02;
  sqrtAsti[2] = 3.6980081924740e+04;
  Bi[2] = 6.9005216444999e-01;
  Fomega[2] = 5.1945301904908e-01;

  // species 3: H2O
  // Imported from NIST
  sqrtOneOverTc[3] = 3.9311140369715e-02;
  sqrtAsti[3] = 1.3145468419592e+05;
  Bi[3] = 1.1727204136223e+00;
  Fomega[3] = 1.0013577274636e+00;

  // species 4: CO
  // Imported from NIST
  sqrtOneOverTc[4] = 8.6759935530451e-02;
  sqrtAsti[4] = 4.3619324712923e+04;
  Bi[4] = 9.7786303823350e-01;
  Fomega[4] = 5.5459948367500e-01;

  // species 5: CO2
  // Imported from NIST
  sqrtOneOverTc[5] = 5.7342616962522e-02;
  sqrtAsti[5] = 4.3745610142919e+04;
  Bi[5] = 6.7506167841745e-01;
  Fomega[5] = 8.2653709187500e-01;

  // species 6: N2
  // Imported from NIST
  sqrtOneOverTc[6] = 8.9019282240563e-02;
  sqrtAsti[6] = 4.2022983929562e+04;
  Bi[6] = 9.5560052221563e-01;
  Fomega[6] = 5.4259343186608e-01;

  // species 7: HE
  // Computed from Lennard-Jones
  sqrtOneOverTc[7] = 2.7294300551596e-01;
  sqrtAsti[7] = 5.4994061413731e+04;
  Bi[7] = 2.1989344021800e+00;
  Fomega[7] = 4.8508000000000e-01;

  // species 8: AR
  // Imported from NIST
  sqrtOneOverTc[8] = 8.1416597423011e-02;
  sqrtAsti[8] = 2.9333346340060e+04;
  Bi[8] = 5.5540880812616e-01;
  Fomega[8] = 4.8197599354800e-01;
}

// gauss-jordan solver external routine
// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve_simplified(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort(
    "sgjsolve_simplified not implemented, choose a different solver ");
}

#endif
