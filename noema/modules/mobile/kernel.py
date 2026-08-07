"""Mobile Module — React Native, Flutter, native app scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MobileFramework(StrEnum):
    REACT_NATIVE = "react_native"
    FLUTTER = "flutter"
    NATIVE_IOS = "native_ios"
    NATIVE_ANDROID = "native_android"
    EXPO = "expo"
    KOTLIN_MULTIPLATFORM = "kotlin_multiplatform"


@dataclass
class Screen:
    name: str = ""
    route: str = ""
    widgets: list[str] = field(default_factory=list)
    has_nav: bool = True
    has_appbar: bool = True


@dataclass
class MobileProject:
    framework: MobileFramework = MobileFramework.REACT_NATIVE
    name: str = ""
    screens: list[Screen] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    state_management: str = ""
    navigation: str = ""


class ProjectGenerator:
    """Generate mobile app project structures."""

    FRAMEWORK_CONFIGS: dict[str, dict[str, str]] = {
        "react_native": {"nav": "react-navigation", "state": "zustand", "ui": "NativeBase"},
        "flutter": {"nav": "go_router", "state": "riverpod", "ui": "Material 3"},
        "expo": {"nav": "expo-router", "state": "zustand", "ui": "React Native Paper"},
        "native_ios": {"nav": "SwiftUI Navigation", "state": "Combine", "ui": "SwiftUI"},
        "native_android": {
            "nav": "Jetpack Navigation",
            "state": "ViewModel",
            "ui": "Material Compose",
        },
    }

    COMMON_SCREENS: dict[str, list[Screen]] = {
        "auth": [
            Screen(
                name="Login",
                route="/login",
                widgets=["email_field", "password_field", "login_button"],
            ),
            Screen(
                name="Register",
                route="/register",
                widgets=["name_field", "email_field", "password_field", "register_button"],
            ),
            Screen(
                name="ForgotPassword",
                route="/forgot-password",
                widgets=["email_field", "reset_button"],
            ),
        ],
        "ecommerce": [
            Screen(name="Home", route="/", widgets=["product_grid", "search_bar", "category_tabs"]),
            Screen(
                name="ProductDetail",
                route="/product/:id",
                widgets=["image_carousel", "add_to_cart", "reviews"],
            ),
            Screen(
                name="Cart", route="/cart", widgets=["cart_items", "checkout_button", "total_price"]
            ),
            Screen(
                name="Checkout",
                route="/checkout",
                widgets=["address_form", "payment_form", "order_summary"],
            ),
            Screen(name="Orders", route="/orders", widgets=["order_list", "order_status"]),
        ],
        "social": [
            Screen(name="Feed", route="/feed", widgets=["post_list", "fab", "pull_to_refresh"]),
            Screen(name="Profile", route="/profile/:id", widgets=["avatar", "bio", "posts_grid"]),
            Screen(name="Messages", route="/messages", widgets=["conversation_list", "search"]),
            Screen(
                name="Chat",
                route="/chat/:id",
                widgets=["message_list", "input_bar", "typing_indicator"],
            ),
        ],
        "dashboard": [
            Screen(
                name="Overview", route="/", widgets=["stats_cards", "charts", "recent_activity"]
            ),
            Screen(
                name="Analytics", route="/analytics", widgets=["line_chart", "bar_chart", "filters"]
            ),
            Screen(
                name="Settings", route="/settings", widgets=["form", "toggle_list", "save_button"]
            ),
        ],
    }

    def generate_project(
        self, name: str, framework: str = "react_native", features: list[str] | None = None
    ) -> MobileProject:
        fw = (
            MobileFramework(framework)
            if framework in [f.value for f in MobileFramework]
            else MobileFramework.REACT_NATIVE
        )
        config = self.FRAMEWORK_CONFIGS.get(framework, self.FRAMEWORK_CONFIGS["react_native"])
        features = features or []

        screens = []
        for feat in features:
            screens.extend(self.COMMON_SCREENS.get(feat, []))

        if not screens:
            screens = [
                Screen(name="Home", route="/", widgets=["list_view", "app_bar"]),
                Screen(name="Detail", route="/detail/:id", widgets=["content_view"]),
            ]

        return MobileProject(
            framework=fw,
            name=name,
            screens=screens,
            features=features,
            state_management=config["state"],
            navigation=config["nav"],
        )

    def generate_flutter_code(self, project: MobileProject) -> str:
        imports = "import 'package:flutter/material.dart';\n"
        if "auth" in project.features:
            imports += "import 'package:firebase_auth/firebase_auth.dart';\n"

        main = (
            "void main() => runApp(const MyApp());\n\n"
            "class MyApp extends StatelessWidget {\n"
            "  const MyApp({super.key});\n\n"
            "  @override\n"
            "  Widget build(BuildContext context) {\n"
            "    return MaterialApp(\n"
            "      title: '" + project.name + "',\n"
            "      theme: ThemeData(useMaterial3: true),\n"
            "      initialRoute: '/',\n"
            "      routes: {\n"
        )
        for screen in project.screens:
            main += (
                "        '" + screen.route + "': (context) => const " + screen.name + "Screen(),\n"
            )
        main += "      },\n    );\n  }\n}\n"

        screens_code = ""
        for screen in project.screens:
            screens_code += "\nclass " + screen.name + "Screen extends StatelessWidget {\n"
            screens_code += "  const " + screen.name + "Screen({super.key});\n\n"
            screens_code += "  @override\n  Widget build(BuildContext context) {\n"
            screens_code += "    return Scaffold(\n"
            if screen.has_appbar:
                screens_code += "      appBar: AppBar(title: const Text('" + screen.name + "')),\n"
            screens_code += "      body: const Center(child: Text('" + screen.name + "')),\n"
            screens_code += "    );\n  }\n}\n"

        return imports + "\n" + main + screens_code

    def generate_rn_code(self, project: MobileProject) -> str:
        imports = "import React from 'react';\n"
        imports += "import { View, Text, StyleSheet } from 'react-native';\n"

        screens = ""
        for screen in project.screens:
            screens += "\nexport function " + screen.name + "Screen() {\n"
            screens += "  return (\n"
            screens += "    <View style={styles.container}>\n"
            screens += "      <Text>" + screen.name + "</Text>\n"
            screens += "    </View>\n"
            screens += "  );\n}\n\n"

        styles = (
            "\nconst styles = StyleSheet.create({\n"
            "  container: { flex: 1, justifyContent: 'center', alignItems: 'center' },\n"
            "});\n"
        )

        return imports + screens + styles


class MobileModule:
    """Standalone mobile module."""

    NAME = "mobile"
    DESCRIPTION = "Mobile app generation: React Native, Flutter, native iOS/Android"

    def __init__(self) -> None:
        self.generator = ProjectGenerator()

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        framework = "react_native"
        for fw in ["flutter", "expo", "native_ios", "native_android"]:
            if fw in tags:
                framework = fw
                break

        features = [
            t for t in tags if t in ("auth", "ecommerce", "social", "dashboard", "messaging")
        ]
        project = self.generator.generate_project(
            name=getattr(task, "title", "App"),
            framework=framework,
            features=features or ["auth"],
        )
        return {
            "type": "mobile",
            "framework": framework,
            "screens": [{"name": s.name, "route": s.route} for s in project.screens],
            "state_management": project.state_management,
            "navigation": project.navigation,
            "features": project.features,
            "_confidence": 0.8,
        }
