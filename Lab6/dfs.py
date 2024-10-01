graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': [],
}
visited = set()


def dfs(ele):
    if ele not in visited:
        print(ele)
        visited.add(ele)
        for node in graph[ele]:
            dfs(node)


dfs('5')
