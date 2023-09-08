import c4d
from c4d import gui
# Welcome to the world of Python

# Main function
def main():
    doc = c4d.documents.GetActiveDocument()
    obj = doc.GetActiveObject()

    indices = [2215, 3886, 4920, 5828]+\
        [2215, 3640, 4801, 5828]+\
        [10455, 11353, 12383,14066]+\
        [10455, 11492, 12653, 14066]

    sel = obj.GetPointS()
    sel.DeselectAll()
    for i in indices:
        sel.Select(i)
    #sel.SetAll(indices)
    obj.Message(c4d.MSG_UPDATE)

    c4d.EventAdd()



# Execute main()
if __name__=='__main__':
    main()