
import os

target_size_mb = 5
target_size_bytes = target_size_mb * 1024 * 1024

path = "maths_big.txt"

def line_for(n: int) -> str:
    s1 = n * (n + 1) // 2
    s2 = n * (n + 1) * (2*n + 1) // 6
    s3 = s1 * s1  # (sum of first n)^2
    return (
        f"n={n}; n^2={n*n}; n^3={n*n*n}; Σ_{{k=1}}^{{n}} k={s1}; Σ_{{k=1}}^{{n}} k^2={s2}; "
        f"(a+b)^{{{n}}} = Σ_{{k=0}}^{{n}} C(n,k)a^{{n-k}}b^{{k}}; ∫_0^1 x^{{{n}}} dx=1/{n+1}; "
        f"Π_{{k=1}}^{{n}} (1+1/k)=n+1; gcd(n,n+1)=1; sin(nπ)=0; "
        f"Euler: e^{{iπ}}+1=0; "
        f"Identity: (n(n+1)/2)^2={s3}; "
        f"Binomial(n,2)={n*(n-1)//2}; Binomial(n,3)={(n*(n-1)*(n-2))//6}\n"
    )

# Write file incrementally until target size
with open(path, "w", encoding="utf-8") as f:
    n = 1
    while True:
        block = []
        for _ in range(2000):
            block.append(line_for(n))
            n += 1
        f.writelines(block)
        if f.tell() >= target_size_bytes:
            break

# Report final size
final_size = os.path.getsize(path)
final_size_mb = final_size / (1024 * 1024)

print(path)
print(f"Final size: {final_size_mb:.2f} MB")