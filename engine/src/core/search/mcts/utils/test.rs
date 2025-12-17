use super::*;

#[test]
fn test_new_root() {
    let root = DoubleLinkedNode::new_root(42);
    assert_eq!(*root.data(), 42);
    assert!(root.parent().is_none());
    assert!(root.children.is_empty());
}

#[test]
fn test_append_child() {
    let mut root = Rc::new(RefCell::new(DoubleLinkedNode::new_root("root")));
    let child = DoubleLinkedNode::append(&mut root, "child");

    // Check child properties
    assert_eq!(*child.borrow().data(), "child");
    assert!(child.borrow().parent().is_some());

    // Check parent's children list
    assert_eq!(root.borrow().children.len(), 1);
    assert!(Rc::ptr_eq(&root.borrow().children[0], &child));

    // Verify parent-child relationship
    let child_parent = child.borrow().parent().unwrap().upgrade().unwrap();
    assert!(Rc::ptr_eq(&root, &child_parent));
}

#[test]
fn test_multiple_children() {
    let mut root = Rc::new(RefCell::new(DoubleLinkedNode::new_root(0)));
    let child1 = DoubleLinkedNode::append(&mut root, 1);
    let child2 = DoubleLinkedNode::append(&mut root, 2);
    let child3 = DoubleLinkedNode::append(&mut root, 3);

    assert_eq!(root.borrow().children.len(), 3);
    assert_eq!(*child1.borrow().data(), 1);
    assert_eq!(*child2.borrow().data(), 2);
    assert_eq!(*child3.borrow().data(), 3);

    // All children should have the same parent
    let parent_upgrade = |child: &Rc<RefCell<DoubleLinkedNode<i32>>>| {
        child.borrow().parent().unwrap().upgrade().unwrap()
    };
    assert!(Rc::ptr_eq(&parent_upgrade(&child1), &root));
    assert!(Rc::ptr_eq(&parent_upgrade(&child2), &root));
    assert!(Rc::ptr_eq(&parent_upgrade(&child3), &root));
}

#[test]
fn test_data_access_and_mutation() {
    let mut node = DoubleLinkedNode::new_root("initial");
    assert_eq!(*node.data(), "initial");

    node.set_data("updated");
    assert_eq!(*node.data(), "updated");
}

#[test]
fn test_try_fold_up_mut_success() {
    // Build tree: root -> child1 -> child2
    let mut root = Rc::new(RefCell::new(DoubleLinkedNode::new_root(1)));
    let mut child1 = DoubleLinkedNode::append(&mut root, 2);
    let child2 = DoubleLinkedNode::append(&mut child1, 3);

    // Test with Option (Try implementation)
    let result =
        DoubleLinkedNode::try_fold_up_mut::<i32, _, Option<i32>>(child2.clone(), 0, |acc, node| {
            Some(acc + *node.borrow().data())
        });

    assert_eq!(result, Some(6)); // 3 + 2 + 1
}

#[test]
fn test_try_fold_up_mut_with_result() {
    // Build tree: root -> child1 -> child2
    let mut root = Rc::new(RefCell::new(DoubleLinkedNode::new_root(10)));
    let mut child1 = DoubleLinkedNode::append(&mut root, 20);
    let child2 = DoubleLinkedNode::append(&mut child1, 30);

    // Test with Result (Try implementation)
    let result = DoubleLinkedNode::try_fold_up_mut::<i32, _, Result<i32, &str>>(
        child2.clone(),
        0,
        |acc, node| {
            let value = *node.borrow().data();
            if value > 25 {
                Err("Value too large")
            } else {
                Ok(acc + value)
            }
        },
    );

    assert_eq!(result, Err("Value too large")); // Fails at child2 with value 30
}

#[test]
fn test_try_fold_up_mut_early_exit() {
    let mut root = Rc::new(RefCell::new(DoubleLinkedNode::new_root(1)));
    let mut child1 = DoubleLinkedNode::append(&mut root, 2);
    let child2 = DoubleLinkedNode::append(&mut child1, 3);

    let result =
        DoubleLinkedNode::try_fold_up_mut::<i32, _, Option<i32>>(child2.clone(), 0, |acc, node| {
            let value = *node.borrow().data();
            if value == 2 {
                None // Early exit
            } else {
                Some(acc + value)
            }
        });

    assert_eq!(result, None);
}

#[test]
fn test_try_fold_up_mut_root_only() {
    let root = Rc::new(RefCell::new(DoubleLinkedNode::new_root(100)));

    let result =
        DoubleLinkedNode::try_fold_up_mut::<i32, _, Option<i32>>(root.clone(), 5, |acc, node| {
            Some(acc + *node.borrow().data())
        });

    assert_eq!(result, Some(105));
}

#[test]
#[should_panic(expected = "Parent was dropped")]
fn test_try_fold_up_mut_dropped_parent() {
    // Create a child whose parent will be dropped
    let mut parent = Rc::new(RefCell::new(DoubleLinkedNode::new_root(1)));
    let child = DoubleLinkedNode::append(&mut parent, 2);

    // Drop parent while keeping child alive
    drop(parent);

    // This should panic when trying to upgrade the weak reference
    let _ = DoubleLinkedNode::try_fold_up_mut::<i32, _, Option<i32>>(child, 0, |acc, node| {
        Some(acc + *node.borrow().data())
    });
}

#[test]
fn test_parent_method() {
    let mut root = Rc::new(RefCell::new(DoubleLinkedNode::new_root("parent")));
    let child = DoubleLinkedNode::append(&mut root, "child");

    // Child should have a parent
    assert!(child.borrow().parent().is_some());

    // Root should not have a parent
    assert!(root.borrow().parent().is_none());

    // Verify the parent is correct
    let parent_weak = child.borrow().parent().unwrap();
    let parent_rc = parent_weak.upgrade().unwrap();
    assert_eq!(*parent_rc.borrow().data(), "parent");
}

#[test]
fn test_new_child_standalone() {
    let parent = Rc::new(RefCell::new(DoubleLinkedNode::new_root("parent")));
    let child = DoubleLinkedNode::new_child(Rc::downgrade(&parent), "child");

    assert_eq!(*child.borrow().data(), "child");
    assert!(child.borrow().parent().is_some());

    // Note: new_child doesn't automatically add to parent's children
    assert_eq!(parent.borrow().children.len(), 0);
}

#[test]
fn test_refcell_interior_mutability() {
    let root = Rc::new(RefCell::new(DoubleLinkedNode::new_root(0)));

    // Mutate through borrow_mut
    root.borrow_mut().set_data(100);
    assert_eq!(*root.borrow().data(), 100);

    // Can have multiple immutable borrows
    let data1 = *root.borrow().data();
    let data2 = *root.borrow().data();
    assert_eq!(data1, 100);
    assert_eq!(data2, 100);
}

#[test]
fn test_complex_tree_structure() {
    // Build a more complex tree:
    //        root
    //       /    \
    //      a      b
    //     / \    /
    //    c   d  e

    let mut root = Rc::new(RefCell::new(DoubleLinkedNode::new_root("root")));
    let mut a = DoubleLinkedNode::append(&mut root, "a");
    let mut b = DoubleLinkedNode::append(&mut root, "b");

    let c = DoubleLinkedNode::append(&mut a, "c");
    let d = DoubleLinkedNode::append(&mut a, "d");
    let e = DoubleLinkedNode::append(&mut b, "e");

    assert_eq!(root.borrow().children.len(), 2);
    assert_eq!(a.borrow().children.len(), 2);
    assert_eq!(b.borrow().children.len(), 1);

    // Test traversal from leaf e
    let result = DoubleLinkedNode::try_fold_up_mut::<String, _, Option<String>>(
        e.clone(),
        String::new(),
        |mut acc, node| {
            acc.push_str(node.borrow().data());
            acc.push('-');
            Some(acc)
        },
    );

    assert_eq!(result, Some("e-b-root-".to_string()));
}
