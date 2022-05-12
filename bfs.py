
import torch

def bfs1(graph, edges, start, end):
    level = 1
    paths = []

    queue = []

    queue.append( graph[0] )
    while queue:

        cur_len = len(queue)

        for k in range( cur_len ):

            path = queue.pop(0)

            node = path[-1]

            if node == end:

                paths.append ( torch.cat(path))#.numpy() )

                continue

            for adjacent in graph[level]:

                if [node,adjacent] not in edges :
                    continue

                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)
        level+=1
    return paths

def find_paths( adj ):


    edges = torch.nonzero(adj).tolist()
    adjT = torch.t(adj)



    tracks = []

    outer=None
    adj_power=[adj]
    adj_T_power=[adjT]
    for i in range(2,35):

        res = torch.matmul( adj_power[-1],adj )
        adj_power.append( res )
        if i == 35-1:
            outer = torch.nonzero(res)
        res = torch.matmul(adj_T_power[-1], adjT)
        adj_T_power.append(res)


    for p1,p2 in outer:

        #if adj_power[-1][p1,p2] > 5:continue

        a = [ matrix[p1,:].bool() for matrix in adj_power ]
        b = [ matrix[p2,:].bool() for matrix in adj_T_power ]

        b.reverse()

        real = [p1[None][None]]
        tracks_prod = []
        for i,j in zip(a,b[1:]):
            hits =torch.nonzero( i&j )
            if tracks_prod == []:
                tracks_prod = hits
            else:
                tracks_prod += torch.cartesian_prod(tracks_prod,hits)
            print(tracks_prod)
            real.append( hits )
        real.append(p2[None][None] )

        tracks += bfs1(real,edges, p1,p2)

    return tracks
