use std::fs;

use deepviewrt::model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rtm = fs::read("model.rtm")?;
    println!("model name: {}", model::name(&rtm)?);
    println!("model inputs: {:?}", model::inputs(&rtm)?);
    println!("model outputs: {:?}", model::outputs(&rtm)?);
    println!("model layer_count: {}", model::layer_count(&rtm));

    for i in 0..model::layer_count(&rtm) {
        println!("model layer {} name: {}", i, model::layer_name(&rtm, i)?);
        // println!("model layer {} type: {}", i, model::layer_type(&rtm, i)?);
    }

    let input_name = model::layer_name(&rtm, model::inputs(&rtm)?[0] as usize)?;
    let input_index = model::layer_lookup(&rtm, input_name)?;
    assert_eq!(input_index, model::inputs(&rtm)?[0] as i32);

    Ok(())
}
