import re

from django import forms

from .services.lip_engine import FINISHES, decode_image

MAX_UPLOAD = 8 * 1024 * 1024


class UploadForm(forms.Form):
    image = forms.FileField()

    def clean_image(self):
        f = self.cleaned_data["image"]
        if f.size > MAX_UPLOAD:
            raise forms.ValidationError("Image must be under 8 MB.")
        try:
            self.decoded = decode_image(f.read())
        except ValueError as e:
            raise forms.ValidationError(str(e))
        return f


class ApplyForm(forms.Form):
    token = forms.CharField(max_length=64)
    color = forms.CharField(max_length=7)
    finish = forms.ChoiceField(choices=[(k, v[0]) for k, v in FINISHES.items()])

    def clean_color(self):
        c = self.cleaned_data["color"].strip()
        if not re.fullmatch(r"#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})", c):
            raise forms.ValidationError("Use a hex colour like #FF0000.")
        return c if c.startswith("#") else "#" + c
