import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from enum import Enum

class ShapeType(Enum):
    # LINE = "line"
    # RECTANGLE = "rectangle"
    BBOX = "bbox"
    HORIZON = "horizon"
    FAULT = "fault"
    # OTHER = "other"

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Drawing Application")
        
        # Initialize variables
        self.current_shape = None
        self.start_x = None
        self.start_y = None
        self.shape_type = ShapeType.BBOX
        self.shapes = []
        self.selected_shape = None
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Variables for connected segments
        self.is_drawing_segments = False
        self.current_segments = []
        self.segment_points = []
        self.last_point = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create toolbar
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Shape selection
        self.shape_var = tk.StringVar(value=ShapeType.BBOX.value)
        for shape in ShapeType:
            ttk.Radiobutton(
                self.toolbar,
                text=shape.value.capitalize(),
                value=shape.value,
                variable=self.shape_var
            ).pack(side=tk.LEFT, padx=5)
        
        # Canvas
        self.canvas = tk.Canvas(
            self.root,
            width=800,
            height=600,
            bg='white'
        )
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Bind events
        self.canvas.bind('<ButtonPress-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        self.canvas.bind('<Button-3>', self.selected_shape)
        
        # Bind Enter key for finishing connected segments
        self.root.bind('<Return>', self.finish_segments)
        
        # Add buttons for additional functionality
        ttk.Button(
            self.toolbar,
            text="Clear All",
            command=self.clear_canvas
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            self.toolbar,
            text="Save",
            command=self.save_drawing
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            self.toolbar,
            text="Load",
            command=self.load_drawing
        ).pack(side=tk.RIGHT, padx=5)

        # Add Undo/Redo buttons
        ttk.Button(
            self.toolbar,
            text="Undo",
            command=self.undo
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            self.toolbar,
            text="Redo",
            command=self.redo
        ).pack(side=tk.RIGHT, padx=5)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())

    def start_drawing(self, event):
        self.shape_type = ShapeType(self.shape_var.get())
        
        if self.shape_type in [ShapeType.HORIZON, ShapeType.FAULT]:
            if not self.is_drawing_segments:
                # Start new sequence of segments
                self.is_drawing_segments = True
                self.segment_points = [(event.x, event.y)]
                self.last_point = (event.x, event.y)
            else:
                # Add new point to existing sequence
                self.segment_points.append((event.x, event.y))
                
                # Draw new segment
                color='red'
                if self.shape_type in [ShapeType.HORIZON]:
                    color='blue'
                line_id = self.canvas.create_line(
                    self.last_point[0], self.last_point[1],
                    event.x, event.y,
                    fill=color, width=2
                )
                self.current_segments.append(line_id)
                self.last_point = (event.x, event.y)
        else:
            # Regular shape drawing
            self.start_x = event.x
            self.start_y = event.y
            
            if self.shape_type in [ShapeType.BBOX]:
                self.current_shape = self.canvas.create_rectangle(
                    self.start_x, self.start_y, event.x, event.y,
                    outline='black', width=2
                )
    
    def draw(self, event):
        if not self.is_drawing_segments and self.current_shape:
            if self.shape_type == ShapeType.BBOX:
                self.canvas.coords(
                    self.current_shape,
                    self.start_x, self.start_y, event.x, event.y
                )
            elif self.shape_type in [ShapeType.RECTANGLE, ShapeType.BBOX]:
                self.canvas.coords(
                    self.current_shape,
                    self.start_x, self.start_y, event.x, event.y
                )
    
    def stop_drawing(self, event):
        if not self.is_drawing_segments and self.current_shape:
            # Store shape data
            shape_data = {
                'type': self.shape_type.value,
                'coords': self.canvas.coords(self.current_shape),
                'id': self.current_shape
            }
            self.shapes.append(shape_data)
            
            # Add to undo stack
            self.add_to_undo_stack('draw', shape_data)
            
            # Add visual indicator for shape type
            x, y = self.canvas.coords(self.current_shape)[:2]
            self.canvas.create_text(
                x, y - 10,
                text=self.shape_type.value.capitalize(),
                fill='blue'
            )
            
            self.current_shape = None

    def finish_segments(self, event):
        if self.is_drawing_segments and len(self.current_segments) > 0:
            # Combine all segments into one shape data
            all_coords = []
            for segment_id in self.current_segments:
                coords = self.canvas.coords(segment_id)
                all_coords.extend(coords)
            
            shape_data = {
                'type': self.shape_type.value,
                'coords': all_coords,
                'id': self.current_segments
            }
            self.shapes.append(shape_data)
            
            # Add to undo stack
            self.add_to_undo_stack('draw', shape_data)
            
            # Add visual indicator for shape type
            x, y = self.canvas.coords(self.current_segments[0])[:2]
            self.canvas.create_text(
                x, y - 10,
                text=self.shape_type.value.capitalize(),
                fill='blue'
            )
            
            # Reset segment drawing state
            self.is_drawing_segments = False
            self.current_segments = []
            self.segment_points = []
            self.last_point = None
    
    def add_to_undo_stack(self, action_type, shape_data):
        """Add an action to the undo stack and clear redo stack"""
        self.undo_stack.append({
            'type': action_type,
            'data': shape_data
        })
        self.redo_stack.clear()  # Clear redo stack when new action is performed
    
    def undo(self):
        """Undo the last drawing action"""
        if not self.undo_stack:
            return
            
        action = self.undo_stack.pop()
        
        if action['type'] == 'draw':
            # Remove the shape from canvas and shapes list
            for shape_id in action['data']['id'] if isinstance(action['data']['id'], list) else [action['data']['id']]:
                self.canvas.delete(shape_id)
            self.shapes.remove(action['data'])
            # Add to redo stack
            self.redo_stack.append(action)
        elif action['type'] == 'clear':
            # Restore all shapes from the backup
            for shape in action['data']:
                if shape['type'] in ['line', 'horizon', 'fault']:
                    coords = shape['coords']
                    if shape['type'] in ['horizon', 'fault']:
                        # Recreate connected segments
                        shape_ids = []
                        for i in range(0, len(coords) - 2, 2):
                            shape_id = self.canvas.create_line(
                                coords[i], coords[i+1],
                                coords[i+2], coords[i+3],
                                fill='black', width=2
                            )
                            shape_ids.append(shape_id)
                        shape['id'] = shape_ids
                    else:
                        shape['id'] = self.canvas.create_line(
                            coords, fill='black', width=2
                        )
                elif shape['type'] in ['rectangle', 'bbox']:
                    shape['id'] = self.canvas.create_rectangle(
                        shape['coords'], outline='black', width=2
                    )
                self.shapes.append(shape)
            self.redo_stack.append(action)
    
    def redo(self):
        """Redo the last undone action"""
        if not self.redo_stack:
            return
            
        action = self.redo_stack.pop()
        
        if action['type'] == 'draw':
            # Redraw the shape
            shape = action['data']
            if shape['type'] in ['line', 'horizon', 'fault']:
                if shape['type'] in ['horizon', 'fault']:
                    coords = shape['coords']
                    shape_ids = []
                    for i in range(0, len(coords) - 2, 2):
                        shape_id = self.canvas.create_line(
                            coords[i], coords[i+1],
                            coords[i+2], coords[i+3],
                            fill='black', width=2
                        )
                        shape_ids.append(shape_id)
                    shape['id'] = shape_ids
                else:
                    shape['id'] = self.canvas.create_line(
                        shape['coords'], fill='black', width=2
                    )
            elif shape['type'] in ['rectangle', 'bbox']:
                shape['id'] = self.canvas.create_rectangle(
                    shape['coords'], outline='black', width=2
                )
            self.shapes.append(shape)
            self.undo_stack.append(action)
        elif action['type'] == 'clear':
            # Clear all shapes again
            self.canvas.delete('all')
            self.shapes.clear()
            self.undo_stack.append(action)
        # Find closest shape to click
        # closest = self.canvas.find_closest(event.x, event.y)
        # if closest:
        #     # Highlight selected shape
        #     if self.selected_shape:
        #         self.canvas.itemconfig(self.selected_shape, fill='black')
        #     self.selected_shape = closest[0]
        #     self.canvas.itemconfig(self.selected_shape, fill='red')
    
    def clear_canvas(self):
        # Store current shapes for undo
        if self.shapes:
            self.add_to_undo_stack('clear', self.shapes.copy())
            
        self.canvas.delete('all')
        self.shapes = []
        self.selected_shape = None
        self.is_drawing_segments = False
        self.current_segments = []
        self.segment_points = []
        self.last_point = None
    
    def save_drawing(self):
        filename = tk.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Drawing As"
        )
        
        if filename:  # If user didn't cancel
            data = []
            for shape in self.shapes:
                data.append({
                    'type': shape['type'],
                    'coords': list(shape['coords'])
                })
            
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to save drawing: {str(e)}")
    
    def load_drawing(self):
        try:
            with open('drawing.json', 'r') as f:
                data = json.load(f)
            
            self.clear_canvas()
            for shape in data:
                if shape['type'] in ['line', 'horizon', 'fault']:
                    # For horizons and faults, create connected segments
                    if shape['type'] in ['horizon', 'fault']:
                        coords = shape['coords']
                        for i in range(0, len(coords) - 2, 2):
                            shape_id = self.canvas.create_line(
                                coords[i], coords[i+1],
                                coords[i+2], coords[i+3],
                                fill='black', width=2
                            )
                    else:
                        shape_id = self.canvas.create_line(
                            shape['coords'],
                            fill='black', width=2
                        )
                elif shape['type'] in ['rectangle', 'bbox']:
                    shape_id = self.canvas.create_rectangle(
                        shape['coords'],
                        outline='black', width=2
                    )
                
                shape_data = {
                    'type': shape['type'],
                    'coords': shape['coords'],
                    'id': shape_id
                }
                self.shapes.append(shape_data)
                
                # Add label
                self.canvas.create_text(
                    shape['coords'][0],
                    shape['coords'][1] - 10,
                    text=shape['type'].capitalize(),
                    fill='blue'
                )
        except FileNotFoundError:
            tk.messagebox.showerror("Error", "No drawing file found.")
        except json.JSONDecodeError:
            tk.messagebox.showerror("Error", "Invalid drawing file format.")
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred while loading the drawing: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()