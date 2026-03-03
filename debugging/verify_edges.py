import csv
import numpy as np

# Check path_length vs euclidean
bad_edges = []
with open('graph_output/edges.csv', newline='') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        try:
            pl = float(row['path_length'])
            eu = float(row['euclidean'])
            if pl < eu - 0.01:  # path_length should be >= euclidean
                bad_edges.append((i, pl, eu, pl-eu))
        except:
            pass

if bad_edges:
    print(f'Found {len(bad_edges)} edges where path_length < euclidean:')
    for i, pl, eu, diff in bad_edges[:10]:
        print(f'  Row {i}: path_length={pl:.2f}, euclidean={eu:.2f}, diff={diff:.2f}')
else:
    print('V All edges have path_length >= euclidean')
    
# Show sample edges
print('\nSample edges:')
with open('graph_output/edges.csv', newline='') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < 10:
            print(f"  {row['u']}-{row['v']}: path={row['path_length']}, euc={row['euclidean']}, tort={row['tortuosity']}")
