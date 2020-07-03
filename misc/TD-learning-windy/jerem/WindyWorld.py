import tkinter as tk
import numpy as np


class Application:
    """
    Windy World Tkinter environment.
    """
    def __init__(self, nb_x, nb_y, init_pos, end, wind):
        self.ipos = init_pos
        self.pos = init_pos
        self.old_pos = init_pos
        self.very_old_pos = init_pos
        self.end = end
        self.wind = wind

        self.steps = 0
        self.steps_id = 0
        self.show = False

        self.simulation = 0
        self.simulation_idx = 0

        self.nb_x = nb_x
        self.nb_y = nb_y
        self.sq_size = 40
        self.offset = 20

        self.case_ids = np.zeros((nb_x, nb_y))

        self.root = tk.Tk()

        self.c = tk.Canvas(self.root, height=600, width=1000, bg='white')

        self.c.bind('<Configure>', self.create_grid)
        self.c.bind("z", self.up)
        self.c.bind("s", self.down)
        self.c.bind("q", self.left)
        self.c.bind("d", self.right)
        self.c.bind("<1>", lambda event: self.c.focus_set())

        self.c.pack(fill=tk.BOTH, expand=True)

        self.fill_grid(self.pos[0], self.pos[1], "blue")
        self.fill_grid(self.end[0], self.end[1], "red")
        self.write_wind()

    def start(self):
        self.show = True
        self.root.mainloop()

    def reset(self):
        self.steps = 0
        self.simulation += 1
        self.pos = self.ipos
        self.old_pos = self.ipos
        self.very_old_pos = self.ipos
        if self.show:
            self.fill_grid(self.pos[0], self.pos[1], "blue")
            self.fill_grid(self.end[0], self.end[1], "red")
        self.write_simulation()

    def move(self, i, j):
        self.write_step()
        self.steps += 1
        self.very_old_pos = self.old_pos
        self.old_pos = self.pos

        j -= self.wind[self.pos[0]]
        self.pos = (min(self.pos[0] + i, self.nb_x - 1) if i > 0 else max(self.pos[0] + i, 0),
                    min(self.pos[1] + j, self.nb_y - 1) if j > 0 else max(self.pos[1] + j, 0))

        if self.show:
            self.fill_grid(self.very_old_pos[0], self.very_old_pos[1], "white")
            self.fill_grid(self.old_pos[0], self.old_pos[1], "SteelBlue1")
            if self.pos[0] == self.end[0] and self.pos[1] == self.end[1]:
                self.fill_grid(self.pos[0], self.pos[1], "yellow")
                self.fill_grid(self.ipos[0], self.ipos[1], "blue")
                self.fill_grid(self.old_pos[0], self.old_pos[1], "white")
            else:
                self.fill_grid(self.pos[0], self.pos[1], "blue")

    def create_grid(self, event=None):
        self.c.delete('grid_line') # Will only remove the grid_line

        # Creates all vertical lines at intevals of 100

        for i in range(0, self.nb_x+1):
            self.c.create_line([(i*self.sq_size + self.offset, self.offset),
                                (i*self.sq_size + self.offset, self.sq_size * self.nb_y + self.offset)],
                               tag='grid_line')

        # Creates all horizontal lines at intevals of 100
        for i in range(0, self.nb_y+1):
            self.c.create_line([(self.offset, i*self.sq_size + self.offset),
                                (self.sq_size * self.nb_x + self.offset, i*self.sq_size + self.offset)],
                               tag='grid_line')

    def fill_grid(self, i, j, color):
        if self.case_ids[i, j] != 0:
            self.c.delete(self.case_ids[i, j])
        end_x = self.offset + (i+1) * self.sq_size
        end_y = self.offset + (j + 1) * self.sq_size
        self.case_ids[i, j] = self.c.create_rectangle(self.offset + i * self.sq_size,
                                                     self.offset + j * self.sq_size,
                                                     end_x, end_y, fill=color, outline="black")

    def up(self, event):
        self.move(0, -1)

    def down(self, event):
        self.move(0, 1)

    def right(self, event):
        self.move(1, 0)

    def left(self, event):
        self.move(-1, 0)

    def write_wind(self):
        for wi in range(len(self.wind)):
            x = self.offset + wi*self.sq_size + self.sq_size//2
            self.c.create_text(x, self.nb_y * self.sq_size + self.offset + 20,
                               anchor=tk.W, font="Arial", text=str(self.wind[wi]))

    def write_step(self):
        self.c.delete(self.steps_id)
        self.steps_id = self.c.create_text(self.offset, self.nb_y * self.sq_size + self.offset + 60,
                           anchor=tk.W, font="Arial", text="Step : " + str(self.steps))

    def write_simulation(self):
        self.c.delete(self.simulation_idx)
        self.simulation_idx = self.c.create_text(
            self.offset, self.nb_y * self.sq_size + self.offset + 100,
            anchor=tk.W, font="Arial", text="Step : " + str(self.simulation))