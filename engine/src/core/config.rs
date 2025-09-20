use std::fmt;

pub enum ConfigOptionType {
    Check {
        default: bool,
        value: bool,
    },
    Spin {
        default: i32,
        min: i32,
        max: i32,
        value: i32,
    },
    Combo {
        default: String,
        options: ComboOptions,
        value: String,
    },
    Button {
        callback: Box<fn()>,
    },
    String(StringType),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StringType(String);

impl Default for StringType {
    fn default() -> Self {
        Self("<empty>".to_string())
    }
}

impl From<&str> for StringType {
    fn from(value: &str) -> Self {
        StringType(value.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComboOptions(pub Vec<String>);

impl fmt::Display for ComboOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for opt in &self.0 {
            write!(f, "var {} ", opt)?;
        }
        Ok(())
    }
}

impl fmt::Display for ConfigOptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigOptionType::Check { default, .. } => {
                write!(f, "check default {default}")
            }
            ConfigOptionType::Spin {
                default, min, max, ..
            } => {
                write!(f, "spin default {default} min {min} max {max}")
            }
            ConfigOptionType::Combo {
                default, options, ..
            } => {
                write!(f, "combo default {default} {options}")
            }
            ConfigOptionType::Button { callback: _ } => {
                write!(f, "button")
            }
            ConfigOptionType::String(string_type) => {
                write!(f, "string default {}", string_type.0)
            }
        }
    }
}

pub struct ConfigOption {
    pub cfg_name: String,
    pub cfg_type: ConfigOptionType,
}

impl fmt::Display for ConfigOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "option name {} type {}", self.cfg_name, self.cfg_type)
    }
}

pub struct Configuration(pub Vec<ConfigOption>);

impl Configuration {
    pub fn find(&self, name: &str) -> Option<&ConfigOption> {
        self.0.iter().find(|opt| opt.cfg_name == name)
    }

    pub fn find_mut(&mut self, name: &str) -> Option<&mut ConfigOption> {
        self.0.iter_mut().find(|opt| opt.cfg_name == name)
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Self(vec![
            ConfigOption {
                cfg_name: String::from("Hash"),
                cfg_type: ConfigOptionType::Spin {
                    default: 16,
                    min: 1,
                    max: 64 * 1024 * 1024,
                    value: 16,
                },
            },
            ConfigOption {
                cfg_name: String::from("Threads"),
                cfg_type: ConfigOptionType::Spin {
                    default: 1,
                    min: 1,
                    max: 1,
                    value: 1,
                },
            },
            ConfigOption {
                cfg_name: String::from("Clear Hash"),
                cfg_type: ConfigOptionType::Button {
                    callback: Box::from(clear_hash as fn()),
                },
            },
        ])
    }
}

pub fn clear_hash() {
    todo!("clear hash");
}
