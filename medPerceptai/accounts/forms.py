from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password

from accounts.models import StaffProfile


class StaffCreationForm(UserCreationForm):
    """Used by Django Admin when adding a new user."""

    role = forms.ChoiceField(
        choices=StaffProfile.Role.choices,
        required=True,
        label="Staff role",
        help_text="Admin can change system settings. Nurse/Doctor only view their unit.",
    )
    assigned_building = forms.CharField(
        max_length=120,
        required=True,
        label="Building name",
        help_text="Must match the camera/stream building for alerts to show.",
    )
    assigned_floor = forms.CharField(
        max_length=30,
        required=True,
        label="Floor number",
        help_text="Example: 3",
    )
    email = forms.EmailField(required=False)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = (
            "username",
            "first_name",
            "last_name",
            "email",
            "role",
            "assigned_building",
            "assigned_floor",
        )

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data.get("email", "")
        user.is_staff = self.cleaned_data.get("role") == StaffProfile.Role.ADMIN

        if commit:
            user.save()
            StaffProfile.objects.update_or_create(
                user=user,
                defaults={
                    "role": self.cleaned_data["role"],
                    "assigned_building": self.cleaned_data["assigned_building"].strip(),
                    "assigned_floor": self.cleaned_data["assigned_floor"].strip(),
                },
            )

        return user


class StaffAccountForm(forms.Form):
    """Create staff login from Settings page (admin only)."""

    username = forms.CharField(
        max_length=150,
        label="Login username",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. nurse_ali"}),
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={"class": "form-control", "autocomplete": "new-password"}),
        label="Password",
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={"class": "form-control", "autocomplete": "new-password"}),
        label="Confirm password",
    )
    first_name = forms.CharField(
        max_length=150,
        required=False,
        label="First name",
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )
    last_name = forms.CharField(
        max_length=150,
        required=False,
        label="Last name",
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )
    email = forms.EmailField(required=False, label="Email", widget=forms.EmailInput(attrs={"class": "form-control"}))
    role = forms.ChoiceField(
        choices=StaffProfile.Role.choices,
        label="Role",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    assigned_building = forms.CharField(
        max_length=120,
        label="Building",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Main Building"}),
    )
    assigned_floor = forms.CharField(
        max_length=30,
        label="Floor",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "3"}),
    )

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("This username is already taken.")
        return username

    def clean(self):
        cleaned = super().clean()
        password1 = cleaned.get("password1")
        password2 = cleaned.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords do not match.")
        if password1:
            validate_password(password1)
        return cleaned

    def save(self):
        user = User.objects.create_user(
            username=self.cleaned_data["username"],
            password=self.cleaned_data["password1"],
            first_name=self.cleaned_data.get("first_name", ""),
            last_name=self.cleaned_data.get("last_name", ""),
            email=self.cleaned_data.get("email", ""),
            is_staff=self.cleaned_data["role"] == StaffProfile.Role.ADMIN,
        )
        StaffProfile.objects.create(
            user=user,
            role=self.cleaned_data["role"],
            assigned_building=self.cleaned_data["assigned_building"].strip(),
            assigned_floor=self.cleaned_data["assigned_floor"].strip(),
        )
        return user
