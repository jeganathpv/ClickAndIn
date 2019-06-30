import Tkinter as tk
import setup as setup
    
from PIL import ImageTk, Image
def buttonOne():
    setup.trainset_code()
    setup.testset_code(test_path)
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()
button = tk.Button(frame, 
                   text="Train And Test", command=buttonOne
                  
                  )
button.pack(side=tk.LEFT)
slogan = tk.Button(frame,
                   text="Train Data",
                   command=buttonOne)
slogan.pack(side=tk.LEFT)
slogan = tk.Button(frame,
                   text="Test Data",
                   command=buttonOne)
slogan.pack(side=tk.LEFT)
slogan = tk.Button(frame,
                   text="WebCam",
                   command=buttonOne)
slogan.pack(side=tk.LEFT)

button = tk.Button(frame, 
                   text="QUIT", 
                   fg="red",
                   command=quit)
button.pack(side=tk.LEFT)



path = 'cq5dam.web.1200.675.jpeg'


img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(root, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
root.mainloop()


