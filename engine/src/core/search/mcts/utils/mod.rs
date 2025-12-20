use std::cell::RefCell;
use std::ops::Try;
use std::rc::Rc;
use std::rc::Weak;

#[cfg(test)]
pub mod test;

#[derive(Default, Debug)]
pub struct DoubleLinkedNode<T> {
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

    pub fn new_child(parent: Weak<RefCell<Self>>, data: T) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            data,
            parent: Some(parent),
            children: vec![],
        }))
    }

    pub fn append(parent: &mut Rc<RefCell<Self>>, data: T) -> Rc<RefCell<Self>> {
        let child = Self::new_child(Rc::downgrade(parent), data);
        parent.borrow_mut().children.push(child.clone());
        child
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn set_data(&mut self, data: T) {
        self.data = data;
    }

    pub fn parent(&self) -> Option<Weak<RefCell<Self>>> {
        self.parent.clone()
    }

    /// Applies `f` to this and all parent nodes, until no more parent is found or `f` returns
    /// residual.
    pub fn try_fold_up_mut<B, F, R>(mut this: Rc<RefCell<Self>>, mut init: B, mut f: F) -> R
    where
        F: FnMut(B, Rc<RefCell<Self>>) -> R,
        R: Try<Output = B>,
    {
        init = f(init, this.clone())?;
        while let Some(parent) = { this.borrow_mut().parent() } {
            if let Some(parent) = Weak::upgrade(&parent) {
                this = parent;
                init = f(init, this.clone())?;
            } else {
                // todo: this this right that we just break when we can't upgrade the parent
                // reference? this would mean that the parent was dropped...
                //break;
                panic!("Parent was dropped")
            }
        }
        R::from_output(init)
    }
}

pub trait IDoubleLinkedNode {
    type Data;

    fn append(parent: &mut Rc<RefCell<Self>>, data: Self::Data) -> Rc<RefCell<Self>>;
}

impl<T> IDoubleLinkedNode for DoubleLinkedNode<T> {
    type Data = T;

    fn append(parent: &mut Rc<RefCell<Self>>, data: Self::Data) -> Rc<RefCell<Self>> {
        Self::append(parent, data)
    }
}
