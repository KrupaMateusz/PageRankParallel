#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

using namespace std;

const double DAMPING = 0.85;
const double EPSILON = 1e-6;
const int MAX_ITER = 100;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Użycie: " << argv[0] << " <plik edges.txt>\n";
        return 1;
    }

    omp_set_num_threads(atoi(argv[2]));

    unordered_map<int, vector<int>> incoming_links;
    unordered_map<int, int> out_degree;
    int max_node = 0;

    ifstream infile(argv[1]);
    string line;
    while (getline(infile, line)) {
        istringstream iss(line);
        int from, to;
        if (!(iss >> from >> to)) continue;
        incoming_links[to].push_back(from);
        out_degree[from]++;
        max_node = max({max_node, from, to});
    }

    int N = max_node + 1;
    vector<double> rank(N, 1.0 / N);
    vector<double> new_rank(N, 0.0);

    auto start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double diff = 0.0;

        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            if (incoming_links.count(i)) {
                for (int src : incoming_links[i]) {
                    sum += rank[src] / out_degree[src];
                }
            }
            new_rank[i] = (1.0 - DAMPING) / N + DAMPING * sum;
            diff += fabs(new_rank[i] - rank[i]);
        }

        rank.swap(new_rank);

        if (diff < EPSILON) break;
    }

    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double>(end - start).count();

    vector<pair<int, double>> ranked;
    for (int i = 0; i < N; ++i)
        ranked.emplace_back(i, rank[i]);

    sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });


    cout << "Liczba użytych wątków: " << omp_get_max_threads() << "\n";
    cout << "Czas działania OpenMP PageRank: " << time << " sekund\n\n";

    cout << "10 największych współczynników PageRank:\n";
    for (int i = 0; i < 10 && i < ranked.size(); ++i)
        cout << "Wierzchołek " << ranked[i].first << ": " << ranked[i].second << "\n";

    cout << "\n10 najmniejszych współczynników PageRank:\n";
    for (int i = ranked.size() - 10; i < ranked.size(); ++i)
        cout << "Wierzchołek " << ranked[i].first << ": " << ranked[i].second << "\n";

    return 0;
}

