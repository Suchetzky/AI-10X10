import tkinter as tk
from tkinter import font as tkFont

class DFSTreeVisualization:
    def __init__(self, root, width=300, height=700):
        self.tree_canvas = tk.Canvas(root, width=width, height=height, bg='white')
        self.tree_canvas.pack(side=tk.RIGHT)
        self.tree_font = tkFont.Font(family="Helvetica", size=10)
        self.node_positions = {}
        self.current_tree_node = None
        self.node_id = 0

    def draw_tree_node(self, node_id, parent_id=None, is_current=False):
        if parent_id is None:
            x, y = 150, 20
        else:
            parent_x, parent_y = self.node_positions[parent_id]
            x, y = parent_x + (node_id * 20), parent_y + 40

        color = 'red' if is_current else 'black'
        self.tree_canvas.create_text(x, y, text=f'Node {node_id}', fill=color, font=self.tree_font)
        self.node_positions[node_id] = (x, y)
        if parent_id is not None:
            parent_x, parent_y = self.node_positions[parent_id]
            self.tree_canvas.create_line(parent_x, parent_y, x, y)

    def update_current_tree_node(self, node_id):
        if self.current_tree_node is not None:
            self.draw_tree_node(self.current_tree_node, is_current=False)
        self.current_tree_node = node_id
        self.draw_tree_node(node_id, is_current=True)

    def reset(self):
        self.tree_canvas.delete('all')
        self.node_positions = {}
        self.current_tree_node = None
        self.node_id = 0
