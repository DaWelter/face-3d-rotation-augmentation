import c4d
from c4d import gui
# Welcome to the world of Python

# Main function
def main():
    doc = c4d.documents.GetActiveDocument()
    obj = doc.GetActiveObject()

    sel = obj.GetPointS()
    items = sel.GetAll(obj.GetPointCount())
    items = [ i for i,b in enumerate(items) if b ]
    print(items)


# Execute main()
if __name__=='__main__':
    main()