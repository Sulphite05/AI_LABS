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
    print(get)
    for i in graph[get]:
        queue.insert(0, i)