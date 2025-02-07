#![allow(dead_code)]
use std::error::Error;
use std::fs::File;
use std::path::Path;

pub fn lookup_table_from_csv(file_path: &str) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    let file = File::open(Path::new(file_path))?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut tuples = Vec::new();

    for result in rdr.records() {
        let record = result?;

        if record.len() != 2 {
            return Err(format!("Expected 2 columns, but found {}", record.len()).into());
        }

        let x: f64 = record[0].parse()?;
        let y: f64 = record[1].parse()?;

        tuples.push((x, y));
    }

    Ok(tuples)
}

pub fn test_math_fn_accuracy(function: fn(f64) -> f64, table_name: &str) -> Vec<(f64, f64)> {
    let reference_array = lookup_table_from_csv(table_name).unwrap();

    let mut errors = vec![];
    for i in 0..reference_array.len() {
        let x = reference_array[i].0;
        let y = reference_array[i].1;

        let y_hat = function(x);
        errors.push((x, (y_hat - y).abs()));
    }
    errors
}

pub fn test_fn_speed(function: fn(f64) -> f64, count: u32) -> f64 {
    let random_values = (0..count).map(|_| rand::random::<f64>()).collect::<Vec<f64>>();

    let start = std::time::Instant::now();
    for x in random_values {
        function(x);
    }
    start.elapsed().as_secs_f64()
}

pub fn plot(path: &str, fns: Vec<Vec<(f64, f64)>>, x_min: f64, x_max: f64, y_min: f64, y_max: f64) {
    use plotlib::page::Page;
    use plotlib::repr::Plot;
    use plotlib::style::{LineStyle, LineJoin};
    use plotlib::view::ContinuousView;

    let colors = vec![
        "red", "blue", "green", "orange", "purple", 
        "cyan", "magenta", "yellow", "brown", "black"
    ];

    let mut plots = Vec::new();

    for (i, fcn) in fns.into_iter().enumerate() {
        let color = colors[i % colors.len()];
        let s: Plot = Plot::new(fcn).line_style(
            LineStyle::new()
                .colour(color)
                .linejoin(LineJoin::Round),
        );
        plots.push(s);
    }

    let mut view = ContinuousView::new();
    for plot in plots {
        view = view.add(plot);
    }

    view = view.x_range(x_min, x_max).y_range(y_min, y_max);

    Page::single(&view).save(path).unwrap();
} 
