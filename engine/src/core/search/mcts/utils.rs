struct DoubleLinkedNode<T> {
    data: T,
    children: Vec<Rc<RefCell<Self>>>,
    parent: Option<Weak<RefCell<Self>>>,
}

impl<T> DoubleLinkedNode<T> {
    pub fn new_root(data: T) -> Self {
        Self {
            data,
            parent: None,
            children: vec![],
        }
    }

    pub fn new_child(parent: Weak<RefCell<EvalInfoNode>>, data: T) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            data,
            parent: Some(parent),
            children: vec![],
        }))
    }

    pub fn append(parent: &mut Rc<RefCell<EvalInfoNode>>, data: T) -> Rc<RefCell<Self>> {
        let child = Self::new_child(parent.downgrade(), data);
        parent.borrow_mut().children.push(child.clone());
        child
    }

    pub fn data(&self) -> &T {
        self.data
    }

    pub fn set_data(&mut self, data: T) {
        self.data = data;
    }
}
