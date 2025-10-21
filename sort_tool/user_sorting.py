from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import os

def display_face_with_list(face, lst, suggestion, crop=None):
    if suggestion and suggestion not in lst:
        raise(ValueError("Suggestion not in input list"))

    def check_input(event):
        value = event.widget.get()

        if value == '':
            combo_box['values'] = lst
        else:
            data = []
            for item in lst:
                if value.lower() in item.lower():
                    data.append(item)

            combo_box['values'] = data

    root = Tk()

    # creating Combobox
    face_name = StringVar()
    if suggestion:
        face_name.set(suggestion)
    combo_box = ttk.Combobox(root, textvariable=face_name)
    combo_box['values'] = lst
    combo_box.bind('<KeyRelease>', check_input)
    combo_box.pack()
    combo_box.focus_set()
    combo_box.select_range(0, END)

    rawimg = Image.open(face)
    if crop:
        rawimg = rawimg.crop((crop['x'], crop['y'], crop['x']+crop['w'], crop['y'] + crop['h']))
    img = ImageTk.PhotoImage(rawimg.resize((500, int(crop["h"]/crop["w"]*500))))
    panel = Label(root, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")

    root.bind("<Return>", lambda _:root.destroy())
    root.mainloop()
    return face_name.get()

if __name__ == "__main__":
    display_face_with_list("public/albums/mic_2025_amiens/100_2001.JPG", ["a", "b"], "a")