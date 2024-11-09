#[const_trait]
pub trait ConstFrom<T> {
    fn from_c(value: T) -> Self;
}

#[const_trait]
pub trait ConstDefault {
    fn default_c() -> Self;
}