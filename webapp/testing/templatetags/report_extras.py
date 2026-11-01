from django import template

register = template.Library()


@register.filter
def pct(value):
    if value is None:
        return 'N/A'
    return f'{value * 100:.2f}'.replace('.', ',') + ' %'


@register.filter
def num(value, digits='3'):
    if value is None:
        return 'N/A'
    return f'{float(value):.{digits}f}'.replace('.', ',')


@register.filter
def pct_of(part, whole):
    if not whole:
        return 'N/A'
    return f'{part / whole * 100:.2f}'.replace('.', ',') + ' %'


@register.filter
def getitem(d, key):
    try:
        return (d or {}).get(key)
    except AttributeError:
        return None


@register.filter
def pval(value):
    """Format p-value; < 0.001 shown as such."""
    if value is None:
        return 'N/A'
    if value < 0.001:
        return '< 0,001'
    return f'{value:.3f}'.replace('.', ',')

