graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': [],
}


def ids(start_node, max_depth):

    def dls(node, depth):
        if depth == 0:
            return [node]
        if node not in graph or depth < 0:
            return []
        result = [node]
        for neighbor in graph[node]:
            result.extend(dls(neighbor, depth - 1))
        return result

    for depth in range(max_depth + 1):
        print(f"Level {depth}:")
        visited_at_depth = set()
        for node in dls(start_node, depth):
            if node not in visited_at_depth:
                print(node)
                visited_at_depth.add(node)


ids('5', 4)
