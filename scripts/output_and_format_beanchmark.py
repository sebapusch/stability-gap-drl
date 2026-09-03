from transformation.benchmarks.transformed_env_benchmark import random_orthogonal, PERMUTATION_SEEDS

BENCHMARK_EASY = [1, 2, 3]
BENCHMARK_HARD = [2, 8, 9]

SIZE = 4

def main() -> None:
    for benchmark in (BENCHMARK_EASY, BENCHMARK_HARD):
        print("=========================")

        for i in benchmark:
            mat, bias = random_orthogonal(
                PERMUTATION_SEEDS[i - 1],
                SIZE,
                i > 5,
            )

            print('\\begin{bmatrix}\\\\')
            for row in mat:
                for col in row:
                    print(f'{round(col, 4)} &')
                print('\\\\')
            print('\\end{bmatrix}')

            if bias is not None:
                print('\\begin{bmatrix}\\\\')
                for entry in bias:
                    print(f'{round(entry, 4)} &')
                print('\\end{bmatrix}')

        print("=========================\n\n")

if __name__ == "__main__":
    main()
