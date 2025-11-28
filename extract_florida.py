import osmium
import csv

class NodeHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.count = 0
    
    def node(self, n):
        if n.location.valid():
            self.nodes.append((n.id, n.location.lon, n.location.lat))
            self.count += 1
            if self.count % 1000000 == 0:
                print(f"Processed {self.count // 1000000}M nodes...")

print("Extracting nodes from Florida OSM data...")
handler = NodeHandler()
handler.apply_file("data/florida-latest.osm.pbf")

print(f"\nTotal nodes: {len(handler.nodes):,}")
print("Writing to CSV...")

with open('data/florida_nodes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'lon', 'lat'])
    for i, (node_id, lon, lat) in enumerate(handler.nodes):
        writer.writerow([node_id, lon, lat])
        if (i + 1) % 1000000 == 0:
            print(f"  Wrote {(i+1) // 1000000}M...")

print("Done! Saved to data/florida_nodes.csv")
