"""Quick stats on edges."""

import csv

edges = []
with open('graph_output/edges.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        edges.append({
            'path_length': float(row['path_length']),
            'euclidean': float(row['euclidean']),
            'tortuosity': float(row['tortuosity'])
        })

print(f'Total edges: {len(edges)}')
print(f'\nPath length (pixels):')
print(f'  Mean: {sum(e["path_length"] for e in edges)/len(edges):.2f}')
print(f'  Min: {min(e["path_length"] for e in edges):.2f}')
print(f'  Max: {max(e["path_length"] for e in edges):.2f}')

print(f'\nEuclidean distance:')
print(f'  Mean: {sum(e["euclidean"] for e in edges)/len(edges):.2f}')
print(f'  Min: {min(e["euclidean"] for e in edges):.2f}')
print(f'  Max: {max(e["euclidean"] for e in edges):.2f}')

print(f'\nTortuosity (path_length / euclidean):')
print(f'  Mean: {sum(e["tortuosity"] for e in edges)/len(edges):.4f}')
print(f'  Min: {min(e["tortuosity"] for e in edges):.4f}')
print(f'  Max: {max(e["tortuosity"] for e in edges):.4f}')

# Check all path >= euclidean
ratio_errors = sum(1 for e in edges if e['path_length'] < e['euclidean'])
print(f'\nEdges where path_length < euclidean: {ratio_errors} ({100*ratio_errors/len(edges):.1f}%)')
