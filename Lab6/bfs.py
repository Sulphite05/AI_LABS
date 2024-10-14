graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': [],
}

queue = ['5']
visited = set()

while queue:
    get = queue.pop()
    if get not in visited:
        print(get)
        visited.add(get)
        for i in graph[get]:
            queue.insert(0, i)