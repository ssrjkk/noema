"""I18n Module — internationalization, localization, translation management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Translation:
    key: str = ""
    locale: str = ""
    value: str = ""
    context: str = ""
    plural_form: str = ""  # singular, plural, few, many
    tags: list[str] = field(default_factory=list)


@dataclass
class LocaleConfig:
    code: str = ""
    name: str = ""
    direction: str = "ltr"
    date_format: str = "YYYY-MM-DD"
    time_format: str = "HH:mm"
    number_separator: str = ","
    decimal_separator: str = "."
    currency: str = "USD"
    currency_symbol: str = "$"


SUPPORTED_LOCALES: dict[str, LocaleConfig] = {
    "en": LocaleConfig(
        code="en", name="English", date_format="MM/DD/YYYY", currency="USD", currency_symbol="$"
    ),
    "ru": LocaleConfig(
        code="ru", name="Russian", date_format="DD.MM.YYYY", currency="RUB", currency_symbol="RUB"
    ),
    "de": LocaleConfig(
        code="de", name="German", date_format="DD.MM.YYYY", currency="EUR", currency_symbol="EUR"
    ),
    "fr": LocaleConfig(
        code="fr", name="French", date_format="DD/MM/YYYY", currency="EUR", currency_symbol="EUR"
    ),
    "ja": LocaleConfig(
        code="ja",
        name="Japanese",
        date_format="YYYY年MM月DD日",
        currency="JPY",
        currency_symbol="¥",
    ),
    "zh": LocaleConfig(
        code="zh", name="Chinese", date_format="YYYY年MM月DD日", currency="CNY", currency_symbol="¥"
    ),
    "ar": LocaleConfig(
        code="ar", name="Arabic", direction="rtl", date_format="DD/MM/YYYY", currency="SAR"
    ),
    "es": LocaleConfig(
        code="es", name="Spanish", date_format="DD/MM/YYYY", currency="EUR", currency_symbol="EUR"
    ),
    "pt": LocaleConfig(
        code="pt", name="Portuguese", date_format="DD/MM/YYYY", currency="BRL", currency_symbol="R$"
    ),
    "ko": LocaleConfig(
        code="ko", name="Korean", date_format="YYYY.MM.DD", currency="KRW", currency_symbol="₩"
    ),
    "hi": LocaleConfig(
        code="hi", name="Hindi", date_format="DD/MM/YYYY", currency="INR", currency_symbol="₹"
    ),
    "uk": LocaleConfig(
        code="uk", name="Ukrainian", date_format="DD.MM.YYYY", currency="UAH", currency_symbol="UAH"
    ),
}


class TranslationStore:
    """In-memory translation store with plurals and interpolation."""

    def __init__(self) -> None:
        self.translations: dict[str, dict[str, Translation]] = {}  # locale -> key -> Translation
        self._namespaces: dict[str, set[str]] = {}  # namespace -> set of keys

    def add(
        self,
        key: str,
        locale: str,
        value: str,
        context: str = "",
        plural_form: str = "",
        tags: list[str] | None = None,
        namespace: str = "common",
    ) -> Translation:
        t = Translation(
            key=key,
            locale=locale,
            value=value,
            context=context,
            plural_form=plural_form,
            tags=tags or [],
        )
        self.translations.setdefault(locale, {})[key] = t
        self._namespaces.setdefault(namespace, set()).add(key)
        return t

    def get(self, key: str, locale: str = "en", count: int | None = None, **kwargs: str) -> str:
        store = self.translations.get(locale, {})
        t = store.get(key)
        if not t:
            store_en = self.translations.get("en", {})
            t = store_en.get(key)
        if not t:
            return key
        value = t.value
        if count is not None:
            if count == 0 and "zero" in store:
                value = store[key].value
            elif count != 1:
                plural_key = key + "_plural"
                if plural_key in store:
                    value = store[plural_key].value
        for k, v in kwargs.items():
            value = value.replace("{" + k + "}", v)
        return value

    def export_locale(self, locale: str) -> dict[str, str]:
        return {key: t.value for key, t in self.translations.get(locale, {}).items()}

    def import_translations(
        self, locale: str, data: dict[str, str], namespace: str = "common"
    ) -> int:
        count = 0
        for key, value in data.items():
            self.add(key, locale, value, namespace=namespace)
            count += 1
        return count

    def missing_keys(self, base_locale: str = "en", target_locale: str = "") -> list[str]:
        if not target_locale:
            return []
        base_keys = set(self.translations.get(base_locale, {}).keys())
        target_keys = set(self.translations.get(target_locale, {}).keys())
        return sorted(base_keys - target_keys)

    def stats(self) -> dict[str, Any]:
        locales = {locale: len(keys) for locale, keys in self.translations.items()}
        return {
            "locales": len(self.translations),
            "total_keys": sum(locales.values()),
            "per_locale": locales,
            "namespaces": len(self._namespaces),
        }


class NumberFormatter:
    @staticmethod
    def format(value: float, locale: str = "en") -> str:
        config = SUPPORTED_LOCALES.get(locale, SUPPORTED_LOCALES["en"])
        if config.decimal_separator == ",":
            formatted = (
                f"{value:,.2f}".replace(",", "X")
                .replace(".", config.decimal_separator)
                .replace("X", config.number_separator)
            )
        else:
            formatted = f"{value:,.2f}".replace(",", config.number_separator)
        return formatted

    @staticmethod
    def format_currency(value: float, locale: str = "en") -> str:
        config = SUPPORTED_LOCALES.get(locale, SUPPORTED_LOCALES["en"])
        num = NumberFormatter.format(value, locale)
        if config.currency_symbol in ("$", "€", "£", "¥", "₹"):
            return config.currency_symbol + num
        return num + " " + config.currency_symbol


class I18nModule:
    """Standalone i18n module."""

    NAME = "i18n"
    DESCRIPTION = "Internationalization, localization, translation management"

    def __init__(self) -> None:
        self.store = TranslationStore()
        self.formatter = NumberFormatter()

    def setup_common_translations(self) -> None:
        common_en = {
            "app.name": "My Application",
            "nav.home": "Home",
            "nav.settings": "Settings",
            "nav.profile": "Profile",
            "nav.logout": "Logout",
            "auth.login": "Log In",
            "auth.register": "Register",
            "auth.forgot_password": "Forgot Password?",
            "auth.email": "Email",
            "auth.password": "Password",
            "common.save": "Save",
            "common.cancel": "Cancel",
            "common.delete": "Delete",
            "common.confirm": "Confirm",
            "common.loading": "Loading...",
            "common.error": "An error occurred",
            "common.not_found": "Not Found",
            "common.items_count": "{count} items",
        }
        common_ru = {
            "app.name": "My Application",
            "nav.home": "Главная",
            "nav.settings": "Настройки",
            "nav.profile": "Профиль",
            "nav.logout": "Выйти",
            "auth.login": "Войти",
            "auth.register": "Регистрация",
            "auth.forgot_password": "Забыли пароль?",
            "auth.email": "Email",
            "auth.password": "Пароль",
            "common.save": "Сохранить",
            "common.cancel": "Отмена",
            "common.delete": "Удалить",
            "common.confirm": "Подтвердить",
            "common.loading": "Загрузка...",
            "common.error": "Произошла ошибка",
            "common.not_found": "Не найдено",
        }
        for key, val in common_en.items():
            self.store.add(key, "en", val)
        for key, val in common_ru.items():
            self.store.add(key, "ru", val)

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        recommended = ["en"]
        for code in SUPPORTED_LOCALES:
            if code in tags:
                recommended.append(code)
        if len(recommended) == 1:
            recommended.extend(["ru", "de", "fr", "ja", "es"])
        return {
            "type": "i18n",
            "recommended_locales": recommended,
            "supported_count": len(SUPPORTED_LOCALES),
            "features": ["plurals", "interpolation", "rtl", "number_formatting", "currency"],
            "_confidence": 0.85,
        }
