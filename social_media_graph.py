import networkx as nx 
import matplotlib.pyplot as plt
import random

# Створення графа
G = nx.Graph()

# Додавання вузлів (людей)
people = range(1, 16)  # 15 людей
G.add_nodes_from(people)

# Додавання ребер (зв'язків між людьми)
connections = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6),
               (5, 6), (5, 7), (6, 8), (7, 8), (8, 9), (8, 10), (9, 10), (9, 11),
               (10, 12), (11, 12), (11, 13), (12, 13), (12, 14), (13, 14), (13, 15), (14, 15)]
G.add_edges_from(connections)

# Алгоритм BFS (Завдання №2)
def bfs(graph, start, goal):
    """ Виконує пошук в ширину (BFS) від start до goal """
    visited = set()
    queue = [[start]]

    if start == goal:
        return [start]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in visited:
            neighbours = graph[node]
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                if neighbour == goal:
                    return new_path
            visited.add(node)

    return []

# Алгоритм DFS (Завдання №2)
def dfs(graph, start, goal, path=[], visited=set()):
    """ Виконує пошук в глибину (DFS) від start до goal """
    path = path + [start]
    visited.add(start)

    if start == goal:
        return path

    for neighbour in graph[start]:
        if neighbour not in visited:
            new_path = dfs(graph, neighbour, goal, path, visited)
            if new_path:
                return new_path
    return []


# Додавання ваг до ребер графа (Завдання №3)
edge_weights = {(1, 2): 3, (1, 3): 5, (2, 3): 2, (2, 4): 4, (3, 4): 1, (3, 5): 2,
                (4, 5): 3, (4, 6): 5, (5, 6): 2, (5, 7): 7, (6, 8): 3, (7, 8): 2,
                (8, 9): 4, (8, 10): 6, (9, 10): 1, (9, 11): 5, (10, 12): 2, (11, 12): 3,
                (11, 13): 4, (12, 13): 1, (12, 14): 2, (13, 14): 3, (13, 15): 6, (14, 15): 4}

for (u, v), weight in edge_weights.items():
    if G.has_edge(u, v):
        G[u][v]['weight'] = weight
    else:
        G.add_edge(u, v, weight=weight)

# Використання алгоритму Дейкстри для знаходження найкоротших шляхів від кожної вершини до всіх інших
shortest_paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

# Перегляд найкоротших шляхів від однієї конкретної вершини (наприклад, від вершини 1)
shortest_paths_from_1 = shortest_paths[1]

# Перетворення графа з NetworkX в словник для зручності використання у функціях
graph_dict = nx.to_dict_of_lists(G)

# Виконання DFS і BFS
start_node = 1
end_node = 15

bfs_path = bfs(graph_dict, start_node, end_node)
dfs_path = dfs(graph_dict, start_node, end_node)

# Візуалізація графа
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G)  # Розташування вершин
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
# Малювання ребер з вагами
nx.draw_networkx_edges(G, pos)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Малювання міток вершин
nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold')

plt.title('Соціальна мережа')
plt.show()

# Основні характеристики графа
number_of_nodes = G.number_of_nodes()
number_of_edges = G.number_of_edges()

average_degree = sum(dict(G.degree()).values()) / number_of_nodes

print(f"кількість вершин: {number_of_nodes}, ребер: {number_of_edges}, ступінь вершин: { average_degree}")
print(f"Шлях, за допомогою BFS: {bfs_path}")
print(f"Шлях, за допомогою DFS: {dfs_path}")
print(f"коротший шлях: {shortest_paths_from_1[end_node]}")  