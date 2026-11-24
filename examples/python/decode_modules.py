"""Run cached self-attention and a SwiGLU MLP on two decode tokens using PyTorch modules."""
import torch
import ndp


def main():
    def projection(inputs, outputs):
        return torch.nn.Linear(inputs, outputs, bias=False, dtype=torch.float16).eval()

    mha = torch.nn.MultiheadAttention(8, 2, batch_first=True, dtype=torch.float16).eval()
    attention = ndp.nn.DecodeSelfAttention(mha, max_tokens=16)
    mlp = ndp.nn.SwiGLU(projection(8, 16), projection(8, 16), projection(16, 8))
    tokens = torch.randn(1, 2, 8, dtype=torch.float16)
    with torch.inference_mode():
        for token in tokens.split(1, dim=1):
            print(mlp(attention(token)))
        attention.reset_history()


if __name__ == "__main__":
    main()
