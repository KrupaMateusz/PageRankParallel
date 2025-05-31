#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

#define THREADS_PER_BLOCK 256
#define DAMPING 0.85
#define EPSILON 1e-6
#define MAX_ITER 100

__global__ void pagerank_kernel(
    int N,
    const int* row_ptr,
    const int* col_idx,
    const int* out_degree,
    const double* rank,
    double* new_rank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
        int src = col_idx[j];
        sum += rank[src] / out_degree[src];
    }
    new_rank[i] = (1.0 - DAMPING) / N + DAMPING * sum;
}

void load_graph_csr(
    const string& filename,
    vector<int>& row_ptr,
    vector<int>& col_idx,
    vector<int>& out_degree,
    int& N)
{
    unordered_map<int, vector<int>> incoming_links;
    unordered_map<int, int> out_deg;

    ifstream infile(filename);
    string line;
    int max_node = 0;

    while (getline(infile, line)) {
        istringstream iss(line);
        int from, to;
        if (!(iss >> from >> to)) continue;
        incoming_links[to].push_back(from);
        out_deg[from]++;
        max_node = max({max_node, from, to});
    }

    N = max_node + 1;
    row_ptr.resize(N + 1, 0);
    out_degree.resize(N, 0);

    for (const auto& [node, links] : incoming_links) {
        row_ptr[node + 1] = links.size();
    }

    for (int i = 1; i <= N; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }

    col_idx.resize(row_ptr[N]);
    vector<int> offset(N, 0);

    for (const auto& [to, froms] : incoming_links) {
        int pos = row_ptr[to];
        for (int src : froms) {
            col_idx[pos + offset[to]] = src;
            offset[to]++;
        }
    }

    for (const auto& [node, deg] : out_deg) {
        out_degree[node] = deg;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Użycie: " << argv[0] << " <plik edges.txt>\n";
        return 1;
    }

    vector<int> row_ptr, col_idx, out_degree;
    int N;
    load_graph_csr(argv[1], row_ptr, col_idx, out_degree, N);

    vector<double> rank(N, 1.0 / N), new_rank(N, 0.0);

    // Alokacja GPU
    int *d_row_ptr, *d_col_idx, *d_out_degree;
    double *d_rank, *d_new_rank;

    cudaMalloc(&d_row_ptr, sizeof(int) * row_ptr.size());
    cudaMalloc(&d_col_idx, sizeof(int) * col_idx.size());
    cudaMalloc(&d_out_degree, sizeof(int) * out_degree.size());
    cudaMalloc(&d_rank, sizeof(double) * N);
    cudaMalloc(&d_new_rank, sizeof(double) * N);

    // Kopiowanie do GPU
    cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(int) * row_ptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), sizeof(int) * col_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_degree, out_degree.data(), sizeof(int) * out_degree.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rank, rank.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        pagerank_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            N, d_row_ptr, d_col_idx, d_out_degree, d_rank, d_new_rank
        );

        swap(d_rank, d_new_rank);
    }

    cudaMemcpy(rank.data(), d_rank, sizeof(double) * N, cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double>(end - start).count();

    // Wypisanie 3 największych i 3 najmniejszych
    vector<pair<int, double>> ranked;
    for (int i = 0; i < N; ++i) ranked.emplace_back(i, rank[i]);

    sort(ranked.begin(), ranked.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    cout << "🕒 Czas działania CUDA PageRank: " << time << " sekund\n\n";
    cout << "n10 największych współczynników PageRank:\n";
    for (int i = 0; i < 10; ++i)
        cout << "Wierzchołek " << ranked[i].first << ": " << ranked[i].second << "\n";

    cout << "\n10 najmniejszych współczynników PageRank:\n";
    for (int i = ranked.size() - 10; i < ranked.size(); ++i)
        cout << "Wierzchołek " << ranked[i].first << ": " << ranked[i].second << "\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_out_degree);
    cudaFree(d_rank);
    cudaFree(d_new_rank);

    return 0;
}
