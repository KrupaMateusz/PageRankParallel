#include <iostream>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <random>
#include <string>

using namespace std;

size_t edge_hash(int u, int v, int N) {
    return static_cast<size_t>(u) * N + v;
}

void generate_graph(int num_vertices, double density) {
    string filename = "edges_" + to_string(num_vertices) + ".txt";
    size_t max_edges = static_cast<size_t>(density * num_vertices * (num_vertices - 1));
    unordered_set<size_t> used;

    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Nie można otworzyć pliku: " << filename << "\n";
        return;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, num_vertices - 1);

    size_t edges_written = 0;
    while (edges_written < max_edges) {
        int u = dist(gen);
        int v = dist(gen);
        if (u == v) continue;

        size_t hash = edge_hash(u, v, num_vertices);
        if (used.count(hash)) continue;

        used.insert(hash);
        outfile << u << " " << v << "\n";
        edges_written++;
    }

    outfile.close();
}

int main() {
    const double density = 0.20;

    for (int v = 5000; v <= 13000; v += 2000) {
        generate_graph(v, density);
        cout << "Wygenerowano plik dla " << v << " wierzchołków\n";
    }

    return 0;
}

