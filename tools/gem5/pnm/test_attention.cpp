// Exercise the exact freestanding entry, including optional bias and FP32 scaling.
#include "attention.cpp"
#include <cmath>
#include <cstdio>
int main() {
  _Float16 x[]{200,200}, w[]{200,200,100,100}, bias[]{2,4}, y[2];
  Arguments a{1,64,{DenseOp::Linear,1,2,2,0.5f},0,x,w,bias,y};
  if (pnm_attention(&a,64) || y[0] != (_Float16)40002 || y[1] != (_Float16)20004) return 1;
  a.shape = {DenseOp::Matmul,1,2,2,0.5f};
  a.bias = nullptr;
  if (pnm_attention(&a,64) || y[0] != 30000 || y[1] != 30000) return 1;
  _Float16 logits[]{-40, -3, 0, 1, 3, 8}, out[6];
  a = {1,64,{DenseOp::Softmax,2,3,0,1},0,logits,nullptr,nullptr,out};
  if (pnm_attention(&a,64)) return 1;
  for (int base = 0; base < 6; base += 3) {
    double sum = 0;
    for (int j=0; j<3; ++j) sum += std::exp(double(logits[base+j]));
    for (int j=0; j<3; ++j)
      if (std::abs(float(out[base+j])-std::exp(double(logits[base+j]))/sum) > .0005) return 1;
  }
  a.shape.op = DenseOp::Transpose;
  if (pnm_attention(&a,64)) return 1;
  for (int i=0; i<2; ++i) for (int j=0; j<3; ++j)
    if (out[j*2+i] != logits[i*3+j]) return 1;
  _Float16 gamma[]{2, 3, 4};
  a.shape = {DenseOp::RMSNorm,2,3,0,1e-5f};
  a.right = gamma;
  if (pnm_attention(&a,64)) return 1;
  for (int row=0; row<2; ++row) {
    float sum = 0;
    for (int j=0; j<3; ++j) sum += float(logits[row*3+j])*float(logits[row*3+j]);
    for (int j=0; j<3; ++j) {
      float expected = float(logits[row*3+j])*float(gamma[j])/std::sqrt(sum/3+1e-5f);
      if (std::abs(float(out[row*3+j])-expected) > .003f) return 1;
    }
  }
  a.shape = {DenseOp::Add,2,3,0,1};
  a.right = logits;
  if (pnm_attention(&a,64)) return 1;
  for (int i=0; i<6; ++i) if (out[i] != 2*float(logits[i])) return 1;
  return pnm_attention(&a,40) == 1 ? 0 : 1;
}
