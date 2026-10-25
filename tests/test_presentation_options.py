from anton.core.tools.generate_artifact import component_page

def test_optional_presentation_features_keep_the_same_editable_program():
    program="await CoworkPage({title:'Operations',sections:[]});"
    plain=component_page.document(program)
    enhanced=component_page.document(program,chart_tables=True,timeline_axis=True,render_contract=True,geo_controls=True)
    assert component_page.existing_program(enhanced)==program
    assert len(enhanced)>len(plain)
    assert 'CoworkPage' in enhanced
