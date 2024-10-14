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
    st = [ele]
    while st:
        get = st.pop()
        if get not in visited:
            st += graph[get][::-1]
            print(get)
            visited.add(get)


dfs('5')

# 5
# 3
# 2
# 4
# 8
# 7
