import pytest
from anton.core.tools.generate_artifact import component_page

def test_presentation_bootstrap_preserves_program_for_followup_edits():
    program="await CoworkPage({title:'Appointments',sections:[]});"
    page=component_page.document(program)
    assert component_page.existing_program(page)==program
    assert '<meta name="viewport"' in page

@pytest.mark.parametrize('program', ['<html></html>', 'await CoworkPage({title:"</script>"});'])
def test_invalid_program_cannot_terminate_the_bootstrap_script(program):
    with pytest.raises(ValueError):
        component_page.document(program)
