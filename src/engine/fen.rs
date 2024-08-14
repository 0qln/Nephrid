pub struct Fen<'parts> { pub v: [&'parts str; 6] }

impl<'parts> TryFrom<&'parts str> for Fen<'parts> {
    type Error = anyhow::Error;

    /// This only validates the number parts of the FEN, not their contents. 
    fn try_from(value: &'parts str) -> Result<Self, Self::Error> {
        let parts = value.split(' ').collect::<Vec<&'parts str>>();
        match parts.len() {
            6 => {
                Ok(Fen { v: [
                    parts[0].into(),
                    parts[1].into(),
                    parts[2].into(),
                    parts[3].into(),
                    parts[4].into(),
                    parts[5].into(),
                ] })
            },
            _ => Err(anyhow::Error::msg("Invalid number of parts")),
        }
    }
}
