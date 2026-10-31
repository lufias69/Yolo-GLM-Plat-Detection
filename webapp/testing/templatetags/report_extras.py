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
