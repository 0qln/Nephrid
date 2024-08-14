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
        options: Vec<String>,
        value: String
    },
    Button {
        callback: Box<fn()>,
    },
    String(Option<String>), // Some -> value, None -> "<empty>"
}

impl From<&ConfigOptionType> for String {
    fn from(val: &ConfigOptionType) -> Self {
        let mut result = String::new();
        
        macro_rules! push_field {
            ($name:expr, $field:expr) => {
                result.push_str($name);
                result.push_str(" ");
                result.push_str(&$field.to_string());
                result.push_str(" ");
            };
        }

        match val {
            ConfigOptionType::Check { default, value } => {
                result.push_str("check ");
                push_field!("default", default);
                push_field!("value", value);
            },
            ConfigOptionType::Spin { default, min, max, value } => {
                result.push_str("spin ");
                push_field!("default", default);
                push_field!("min", min);
                push_field!("max", max);
                push_field!("value", value);
            },
            ConfigOptionType::Combo { default, options, value } => {
                result.push_str("combo ");
                push_field!("default", default);
                for opt in options {
                    push_field!("var", opt);
                }
                push_field!("value", value);
            },
            ConfigOptionType::Button { callback: _ } => {
                result.push_str("button ");
            },
            ConfigOptionType::String(maybe) => {
                result.push_str("string");
                push_field!("value", &match maybe {
                    Some(str) => str.clone(),
                    None => String::from("<empty>")
                });
            },
        };
        result
    }
}

pub struct ConfigOption {
    pub cfg_name: String,
    pub cfg_type: ConfigOptionType,
}

impl From<&ConfigOption> for String {
    fn from(option: &ConfigOption) -> String {
        let type_str: String = (&option.cfg_type).into();
        let mut result = String::from("option");
        result.push_str(" name ");
        result.push_str(&option.cfg_name);
        result.push_str(" type ");
        result.push_str(&type_str);
        result
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


