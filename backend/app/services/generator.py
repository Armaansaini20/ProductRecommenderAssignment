from typing import Dict

def generate_description(meta: Dict, style: str = "concise") -> str:
    """
    Template-based description generator.
    meta: product metadata dict with keys like title, description, price, category.
    style: "concise" | "friendly" | "sales"
    """
    title = meta.get("title", "")
    base = meta.get("description", "")
    price = meta.get("price")
    category = meta.get("category", "product")

    if style == "friendly":
        template = f"Meet the {title} — {base} Perfect for anyone who wants a reliable {category}."
        if price:
            template += f" Priced at ${price:.2f}, it’s a solid pick."
        return template

    if style == "sales":
        template = f"{title}: {base} Buy now and upgrade your {category} experience!"
        if price:
            template += f" Only ${price:.2f}."
        return template

    # default concise
    template = f"{title} — {base}"
    if price:
        template += f" (${price:.2f})"
    return template
