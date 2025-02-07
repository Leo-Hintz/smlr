#!/usr/bin/env wolframscript

(* Define the function *)
f[x_?NumericQ] := N[1.0 / (1.0 + Exp[-x]), 308];

(* Create the lookup table for integer inputs from 0 to 10 *)
lookupTable = Table[{x, f[x]}, {x, -700.0, 700.0, 0.1}];

(* Export the lookup table as a CSV file *)
Export["lookup_tables/sigmoid_lookup.csv", lookupTable, "CSV"]