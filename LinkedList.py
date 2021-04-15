class Node:

    # Function to initialise the node object
    def __init__(self, data):
        self.data = data  # Assign data
        self.next = None  # Initialize next as null


# Linked List class contains a Node object
class LinkedList:

    # Function to initialize head
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None and hasattr(node, 'data'):
            nodes.append(node.data)
            node = node.next
        nodes.append("None")
        return " -> ".join([str(node) for node in nodes])

    def __len__(self):
        return self.length

    # This function is defined in Linked List class
    # Appends a new node at the end.  This method is
    #  defined inside LinkedList class shown above */
    def append(self, new_data):
        # 1. Create a new node
        # 2. Put in the data
        # 3. Set next as None
        new_node = Node(new_data)

        self.length += 1

        # 4. If the Linked List is empty, then make the
        #    new node as head
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            return

        # 5. Else go to the last node
        last = self.tail

        # 6. Change the next of last node
        last.next = new_node

        # 7. Set tail as new node
        self.tail = new_node

    def to_list(self):
        ret = []

        node = self.head
        while node is not None and hasattr(node, 'data'):
            ret.append(node.data)
            node = node.next

        return ret

    @staticmethod
    def from_list(arr):
        ll = LinkedList()
        for x in arr:
            ll.append(x)
        return ll
