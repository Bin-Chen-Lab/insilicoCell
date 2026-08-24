A list of APIs in InsilicoCell (Claude/Codex interface):

| API | Purpose | Input | Output |
| --- | --- | --- | --- |
| `get_insilicocell_capabilities` | Discover tasks and schemas | None | Tasks, fields, units, interpretations, and limits |
| `get_insilicocell_resource_limits` | Report current limits | None | Upload, runtime, retention, concurrency, and batch limits |
| `get_insilicocell_welcome` | Provide first-use guidance | Client context | Tutorial, limitations, and compound library choices |
| `preflight_insilicocell_run` | Expand workload and estimate runtime | Task names and dimensional counts | Accepted or refused state and task-specific estimate |
| `validate_insilicocell_inputs` | Validate a prediction request | Task, samples, and optional transcriptome reference | Validity, issues, and context provenance |
| `predict_drug_induced_gene_expression` | Predict drug-induced expression change | SMILES, cell, gene, time, and dose | Z-score |
| `predict_drug_sensitivity` | Predict cell-growth inhibition | SMILES and cell | AUC |
| `predict_drug_protein_binding` | Predict binding affinity | SMILES and amino acid sequence | log<sub>10</sub>(IC<sub>50</sub> in nM) |
| `predict_gene_effect_score` | Predict gene dependency | Gene and cell | Dependency score |
| `predict_gene_mutation` | Predict mutation status | Gene and cell | Mutation probability |
| `predict_cnv` | Predict copy-number variation | Gene and cell | log<sub>2</sub>(copy number + 1) |
| `predict_tf_gene_association` | Predict TF–gene association | Gene and amino acid sequence | Association probability |
| `get_external_input_workflow` | Explain missing-input procedures | None | Acquisition choices and file specifications |
| `begin_data_upload` | Start a small Base64 upload | Data type and filename | Upload identifier |
| `append_data_upload` | Append small-file chunks | Upload ID and Base64 chunk | Progress or validated data reference |
| `begin_direct_file_upload` | Start a direct or browser upload | Data type and filename | One-time upload URL |
| `get_data_upload_status` | Check direct-upload state | Upload ID | Progress or validated data reference |
| `import_data_from_url` | Import authorized public data | Data type and public HTTPS URL | Validated data reference and provenance |
| `delete_temporary_insilicocell_data` | Delete a staged input immediately | Data reference | Deletion status |
| `cleanup_expired_insilicocell_data` | Enforce retention policies | None | Deleted objects and reclaimed storage |
| `get_insilicocell_installation_status` | Check deployment readiness | None | Runtime, checkpoint, and expression-matrix status |
| `install_insilicocell_assets` | Configure a local deployment | Client context | Progress and installation state |
| `get_large_compound_library_options` | Present library choices | None | Enamine or upload options |
| `start_large_compound_screen` | Launch an asynchronous library screen | Compound library, endpoint, and endpoint-specific inputs | Job identifier and preflight record |
| `get_large_screen_status` | Monitor an asynchronous screen | Job ID | Progress, completion, and download information |
| `screen_compounds_by_criteria` | Apply explicit biological thresholds | Compound library, target, cell, and thresholds | Compounds matching the user-selected criteria |
