#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std;

const double DAMPING = 0.85;
const double EPSILON = 1e-6;
const int MAX_ITER = 100;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Podaj nazwę pliku z krawędziami\n";
        return 1;
    }

    ifstream infile(argv[1]);
    if (!infile.is_open()) {
        cerr << "Nie można otworzyć pliku: " << argv[1] << "\n";
        return 1;
    }

    string line;
    int max_node = 0;
    unordered_map<int, vector<int>> incoming_links;
    unordered_map<int, int> out_degree;

    // Wczytywanie grafu
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

    // Pomiar czasu start
    auto start_time = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double diff = 0.0;
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int src : incoming_links[i]) {
                sum += rank[src] / out_degree[src];
            }
            new_rank[i] = (1 - DAMPING) / N + DAMPING * sum;
            diff += fabs(new_rank[i] - rank[i]);
        }
        rank.swap(new_rank);
        if (diff < EPSILON) break;
    }

    auto end_time = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double>(end_time - start_time).count();

    // Sortowanie i wyświetlanie top/min PageRank
    vector<pair<int, double>> node_ranks;
    for (int i = 0; i < N; ++i) {
        node_ranks.emplace_back(i, rank[i]);
    }

    sort(node_ranks.begin(), node_ranks.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second; // malejąco
              });

    cout << "Czas obliczeń PageRank: " << duration << " sekund\n\n";

    cout << "10 największych współczynników PageRank:\n";
    for (int i = 0; i < 10 && i < node_ranks.size(); ++i) {
        cout << "Wierzchołek " << node_ranks[i].first
                  << " z PageRankiem " << node_ranks[i].second << "\n";
    }

    cout << "\n10 najmniejszych współczynników PageRank:\n";
    for (int i = node_ranks.size() - 10; i < node_ranks.size(); ++i) {
        cout << "Wierzchołek " << node_ranks[i].first
                  << " z PageRankiem " << node_ranks[i].second << "\n";
    }

    return 0;
}

